import torch
from torch import nn, optim


from .aknn import BGNNA, BGCNA, Transformer
from .metric_fn import get_metrics, get_metrics_fpr
from .model_help import BaseModel
from .dataset import PairGraphData
from . import MODEL_REGISTRY

import tensorflow as tf
from scipy.sparse import csr_matrix
import community as cm
import networkx as nx
import numpy as np
import scipy.sparse as sp

from torch import Tensor
from torch.nn import Linear
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing

import torch.nn.functional as F
from torch_geometric.nn import GATConv

class ImprovedNeighborEmbedding(nn.Module):
    def __init__(self, num_embeddings, out_channels=128, dropout=0.5, cached=True, bias=True, lamda=0.8, share=True):
        super(ImprovedNeighborEmbedding, self).__init__()

        self.shutcut = nn.Linear(in_features=num_embeddings, out_features=out_channels)

        self.gcn = GATConv(in_channels=num_embeddings, out_channels=out_channels, dropout=dropout)

        self.gcn2 = Transformer(num_embeddings, num_embeddings, 32, 1, 0.1)

        self.dropout = nn.Dropout(dropout)
        self.output_dim = out_channels

    def forward(self, x, edge, embedding):

        from torch_geometric.data import Data
        dynamic_data = Data(x=embedding, edge_index=edge)
        x = F.relu(self.gcn(dynamic_data.x, dynamic_data.edge_index[0]))
        drug_embedding = self.gcn2(embedding, edge[0])
        x = x + drug_embedding
        x = self.dropout(x)
        x = F.normalize(x)

        return x


class NeighborEmbedding(nn.Module):
    def __init__(self, num_embeddings, out_channels=128, dropout=0.5, cached=True, bias=True, lamda=0.8, share=True):
        super(NeighborEmbedding, self).__init__()

        self.shutcut = nn.Linear(in_features=num_embeddings, out_features=out_channels)

        self.bgnn = BGCNA(in_channels=num_embeddings, out_channels=out_channels,
                          cached=cached, bias=bias, lamda=lamda, share=share)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = out_channels
        self.gcn2 = Transformer(num_embeddings, num_embeddings, 32, 1, 0.1)

    def forward(self, x, edge, embedding):
        if not hasattr(self, "edge_index"):
            edge_index = torch.sparse_coo_tensor(*edge)
            self.register_buffer("edge_index", edge_index)
        edge_index = self.edge_index

        drug_embedding = self.gcn2(embedding, edge[0])
        embedding = self.bgnn(embedding, edge_index=edge_index)
        embedding = drug_embedding+embedding

        embedding = self.dropout(embedding)
        x = F.embedding(x, embedding)
        x = F.normalize(x)

        return x


class InteractionEmbedding(nn.Module):
    def __init__(self, n_drug, n_disease, embedding_dim, dropout=0.5):
        super(InteractionEmbedding, self).__init__()
        self.drug_project = nn.Linear(n_drug, embedding_dim, bias=False)
        self.disease_project = nn.Linear(n_disease, embedding_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.output_dim = embedding_dim

        self.vars = {}
        self.vars['weights'] = weight_variable_glorot(
                n_drug, n_drug, name='weights')
        self.vars['weights2'] = weight_variable_glorot(
            n_disease, n_disease, name='weights2')

    def forward(self, association_pairs, drug_embedding, disease_embedding):
        drug_embedding = torch.diag(torch.ones(drug_embedding.shape[0], device=drug_embedding.device))
        disease_embedding = torch.diag(torch.ones(disease_embedding.shape[0], device=disease_embedding.device))

        drug_embedding = self.drug_project(drug_embedding)
        disease_embedding = self.disease_project(disease_embedding)

        drug_embedding = F.embedding(association_pairs[0,:], drug_embedding)
        disease_embedding = F.embedding(association_pairs[1,:], disease_embedding)

        associations = drug_embedding*disease_embedding

        associations = F.normalize(associations)
        associations = self.dropout(associations)


        return associations
# 改进后的interactionDecoder
class InteractionDecoderImproved(nn.Module):
    def __init__(self, in_channels, hidden_dims=(256, 64), out_channels=1, dropout=0.5):
        super(InteractionDecoderImproved, self).__init__()
        decoder = []
        in_dims = [in_channels]+list(hidden_dims)
        out_dims = hidden_dims

        for in_dim, out_dim in zip(in_dims, out_dims):
            decoder.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            decoder.append(nn.LeakyReLU(0.2))

        decoder.append(nn.Linear(hidden_dims[-1], out_channels))
        decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        return self.decoder(x)


class InteractionDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dims=(256, 64), out_channels=1, dropout=0.5):
        super(InteractionDecoder, self).__init__()
        decoder = []
        in_dims = [in_channels]+list(hidden_dims)
        out_dims = hidden_dims
        for in_dim, out_dim in zip(in_dims, out_dims):
            decoder.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            decoder.append(nn.ReLU(inplace=True))
            # decoder.append(nn.Dropout(dropout))
        decoder.append(nn.Linear(hidden_dims[-1], out_channels))
        decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        return self.decoder(x)


@MODEL_REGISTRY.register()
class WMAGT(BaseModel):
    DATASET_TYPE = "PairGraphDataset"
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("WMAGT model config")
        parser.add_argument("--embedding_dim", default=64, type=int, help="编码器关联嵌入特征维度")
        parser.add_argument("--neighbor_embedding_dim", default=32, type=int, help="编码器邻居特征维度")
        parser.add_argument("--hidden_dims", type=int, default=(64, 32), nargs="+", help="解码器每层隐藏单元数")
        parser.add_argument("--lr", type=float, default=5e-4)
        parser.add_argument("--dropout", type=float, default=0.4)
        parser.add_argument("--pos_weight", type=float, default=1.0, help="no used, overwrited, use for bce loss")
        parser.add_argument("--alpha", type=float, default=0.5, help="use for focal loss")
        parser.add_argument("--gamma", type=float, default=2.0, help="use for focal loss")
        parser.add_argument("--lamda", type=float, default=0.8, help="weight for bgnn")
        parser.add_argument("--loss_fn", type=str, default="focal", choices=["bce", "focal"])
        parser.add_argument("--separate", default=False, action="store_true")
        return parent_parser
    # dropout=0.3  lamda=0.8
    def __init__(self, n_drug, n_disease,drug_edge_len, embedding_dim=64, neighbor_embedding_dim=32, hidden_dims=(64, 32),
                 lr=5e-4, dropout=0.3, pos_weight=1.0, alpha=0.5, gamma=2.0, lamda=0.8,
                 loss_fn="focal", separate=False, **config):
        super(WMAGT, self).__init__()
        # lr=0.1
        self.n_drug = n_drug
        self.n_disease = n_disease
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims

        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("gamma", torch.tensor(gamma))
        "rank bce mse focal"
        self.loss_fn_name = loss_fn
        share = not separate
        self.drug_neighbor_encoder = NeighborEmbedding(num_embeddings=n_drug,
                                                       out_channels=neighbor_embedding_dim,
                                                       dropout=dropout, lamda=lamda, share=share)
        self.drug_neighbor_encoder2 = ImprovedNeighborEmbedding(num_embeddings=n_drug,
                                                       out_channels=neighbor_embedding_dim,
                                                       dropout=dropout, lamda=lamda, share=share)

        self.disease_neighbor_encoder2 = ImprovedNeighborEmbedding(num_embeddings=n_disease,
                                                          out_channels=neighbor_embedding_dim,
                                                          dropout=dropout, lamda=lamda, share=share)
        self.disease_neighbor_encoder = NeighborEmbedding(num_embeddings=n_disease,
                                                          out_channels=neighbor_embedding_dim,
                                                          dropout=dropout, lamda=lamda, share=share)
        self.interaction_encoder = InteractionEmbedding(n_drug=n_drug, n_disease=n_disease,
                                                        embedding_dim=embedding_dim, dropout=dropout)

        merged_dim = self.disease_neighbor_encoder.output_dim\
                    +self.interaction_encoder.output_dim  +self.drug_neighbor_encoder.output_dim+self.interaction_encoder.output_dim

        self.decoder = InteractionDecoder(in_channels=merged_dim, hidden_dims=hidden_dims, dropout=dropout,
                                          )
        self.decoder2 = InteractionDecoderImproved(in_channels=merged_dim, hidden_dims=hidden_dims, dropout=dropout,
                                          )

        self.config = config
        self.lr = lr
        self.save_hyperparameters()

        self.gcn2 = Transformer(n_drug, n_drug, n_drug, 1, 0.1)

        self.gcn3 = GraphSAGE(drug_edge_len, n_drug, n_drug, 1, 0.1)

        self.weight2drug = nn.Parameter(torch.Tensor(32, 32))
        self.weight2disease = nn.Parameter(torch.Tensor(32, 32))

        self.reset_parameters()
        self.att = tf.Variable(tf.constant([0.5, 0.33, 0.25]))
        self.vars = {}
        self.vars['weights'] = weight_variable_glorot(
                n_drug, n_drug, name='weights')
        self.vars['weights2'] = weight_variable_glorot(
            n_disease, n_disease, name='weights2')

    def forward(self, interaction_pairs, drug_edge, disease_edge, drug_embedding, disease_embedding):


        drug_embedding = self.gcn2(drug_embedding, drug_edge[0])

        # drug_neighbor_embedding = self.drug_neighbor_encoder(interaction_pairs[0, :], drug_edge_norm, drug_embedding)
        drug_neighbor_embedding = self.drug_neighbor_encoder(interaction_pairs[0,:], drug_edge, drug_embedding)
        disease_neighbor_embedding = self.disease_neighbor_encoder(interaction_pairs[1,:], disease_edge, disease_embedding)
        # disease_neighbor_embedding = self.disease_neighbor_encoder(interaction_pairs[1,:], disease_edge, disease_embedding)
        interaction_embedding = self.interaction_encoder(interaction_pairs, drug_embedding, disease_embedding)


        embedding = torch.cat([drug_neighbor_embedding, interaction_embedding,disease_neighbor_embedding,interaction_embedding], dim=-1)
        score = self.decoder(embedding)
        # score = self.decoder(interaction_embedding)
        return score.reshape(-1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight2drug)
        nn.init.xavier_uniform_(self.weight2disease)
        # self._cached_edge_index = None
        # self._cached_adj_t = None

    def loss_fn(self, predict, label, u, v, u_edge, v_edge, reduction="sum"):
        bce_loss = self.bce_loss_fn(predict, label, self.pos_weight)
        focal_loss = self.focal_loss_fn(predict, label, gamma=self.gamma, alpha=self.alpha)
        mse_loss = self.mse_loss_fn(predict, label, self.pos_weight)
        rank_loss = self.rank_loss_fn(predict, label)

        u_graph_loss = self.graph_loss_fn(x=u, edge=u_edge, cache_name="ul",
                                          # topk=5,
                                          topk = self.config["drug_neighbor_num"],
                                          reduction=reduction)
        v_graph_loss = self.graph_loss_fn(x=v, edge=v_edge, cache_name="vl",
                                          # topk=5,
                                          topk = self.config["disease_neighbor_num"],
                                          reduction=reduction)
        graph_loss = u_graph_loss * self.lambda1 + v_graph_loss * self.lambda2


        loss = {}
        loss.update(bce_loss)
        loss.update(focal_loss)
        loss.update(mse_loss)
        loss.update(rank_loss)
        loss["loss_graph"] = graph_loss
        loss["loss_graph_u"] = u_graph_loss
        loss["loss_graph_v"] = v_graph_loss
        loss["loss"] = loss[f"loss_{self.loss_fn_name}"]+graph_loss
        return loss


    def step(self, batch:PairGraphData):
        interaction_pairs = batch.interaction_pair
        label = batch.label
        drug_edge = batch.u_edge
        disease_edge = batch.v_edge
        drug_embedding = batch.u_embedding
        disease_embedding = batch.v_embedding
        u = self.interaction_encoder.drug_project.weight.T
        v = self.interaction_encoder.disease_project.weight.T

        predict = self.forward(interaction_pairs, drug_edge, disease_edge, drug_embedding, disease_embedding)
        if not self.training:
            predict = predict[batch.valid_mask.reshape(*predict.shape)]
            label = label[batch.valid_mask]
        ans = self.loss_fn(predict=predict, label=label, u=u, v=v, u_edge=drug_edge, v_edge=disease_edge)
        ans["predict"] = predict
        ans["label"] = label
        return ans


    def training_step(self, batch, batch_idx=None):
        return self.step(batch)



    def validation_step(self, batch, batch_idx=None):

        interaction_pairs = batch.interaction_pair
        label = batch.label
        drug_edge = batch.u_edge
        disease_edge = batch.v_edge
        drug_embedding = batch.u_embedding
        disease_embedding = batch.v_embedding
        u = self.interaction_encoder.drug_project.weight.T
        v = self.interaction_encoder.disease_project.weight.T

        predict = self.forward(interaction_pairs, drug_edge, disease_edge, drug_embedding, disease_embedding)

        return self.step(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(lr=self.lr, params=self.parameters(), weight_decay=1e-4)
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.05*self.lr, max_lr=self.lr,
                                                   gamma=0.95, mode="exp_range", step_size_up=4,
                                                   cycle_momentum=False)
        return [optimizer], [lr_scheduler]

    @property
    def lambda1(self):
        max_value = 0.125
        value = self.current_epoch/18.0*max_value
        return torch.tensor(value, device=self.device)

    @property
    def lambda2(self):
        max_value = 0.0625
        value = self.current_epoch / 18.0 * max_value
        return torch.tensor(value, device=self.device)


def calculate_tpr(outputs, targets):
    tpr = get_metrics_fpr(real_score=targets, predict_score=outputs)  # 计算 TPR 的代码
    return tpr

class InnerProductDecoder():
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim, name, num_r, dropout=0., act=tf.nn.sigmoid):
        self.name = name
        self.vars = {}
        self.issparse = False
        self.dropout = dropout
        self.act = act
        self.num_r = num_r
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, input_dim, name='weights')

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            # inputs = tf.nn.dropout(inputs, 1-self.dropout)
            R = inputs[0:self.num_r, :]
            D = inputs[self.num_r:, :]
            R = tf.matmul(R, self.vars['weights'])
            D = tf.transpose(D)
            x = tf.matmul(R, D)
            x = tf.reshape(x, [-1])
            outputs = self.act(x)
        return outputs

import numpy as np
def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = tf.compat.v1.random_uniform(
        [input_dim, output_dim],
        minval=-init_range,
        maxval=init_range,
        dtype=tf.float32
    )

    return tf.Variable(initial, name=name)


def sparsification(adj_louvain, s=1):

        # Number of nodes
    n = adj_louvain.shape[0]

        # Compute degrees
    degrees = np.sum(adj_louvain, axis=0).getA1()

    for i in range(n):

            # Get non-null neighbors of i
        edges = sp.find(adj_louvain[i, :])[1]

            # More than s neighbors? Subsample among those with degree > s
        if len(edges) > s:
                # Neighbors of i with degree > s
            high_degrees = np.where(degrees > s)
            edges_s = np.intersect1d(edges, high_degrees)
                # Keep s of them (if possible), randomly selected
            removed_edges = np.random.choice(edges_s, min(len(edges_s), len(edges) - s), replace=False)
            adj_louvain[i, removed_edges] = 0.0
            adj_louvain[removed_edges, i] = 0.0
            degrees[i] = s
            degrees[removed_edges] -= 1

    adj_louvain.eliminate_zeros()

    return adj_louvain

def louvain_clustering(adj, s_rec):

        # Community detection using the Louvain method
    partition = cm.best_partition(nx.from_scipy_sparse_matrix(adj))
    communities_louvain = list(partition.values())

        # Number of communities found by the Louvain method
    nb_communities_louvain = np.max(communities_louvain) + 1

        # One-hot representation of communities
    communities_louvain_onehot = sp.csr_matrix(np.eye(nb_communities_louvain)[communities_louvain])

        # Community membership matrix (adj_louvain[i,j] = 1 if nodes i and j are in the same community)
    adj_louvain = communities_louvain_onehot.dot(communities_louvain_onehot.transpose())

        # Remove the diagonal
    adj_louvain = adj_louvain - sp.eye(adj_louvain.shape[0])

        # s-regular sparsification of adj_louvain
    adj_louvain = sparsification(adj_louvain, s_rec)

    return adj_louvain, nb_communities_louvain, partition



def preprocess_graph(adj):

    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])

    degree_mat_inv_sqrt = sp.diags(np.power(np.array(adj_.sum(1)), -0.5).flatten())

    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)

    return sparse_to_tuple(adj_normalized)



def sparse_to_tuple(sparse_mx):

    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()

    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape

    return coords, values, shape

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GraphSAGE,self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, edge_attr, emb_ea):
        edge_attr = torch.mm(edge_attr, emb_ea)
        for conv in self.convs[:-1]:
            x = conv(x, adj_t, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.convs[-1](x, adj_t, edge_attr)
        return x


class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(out_channels, out_channels, bias=False)
            # self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)


        out = self.propagate(edge_index, x=edge_attr, edge_attr=edge_attr, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out


    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return F.relu(x_j + edge_attr)


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

