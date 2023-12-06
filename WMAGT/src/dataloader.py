import logging
import os
import torch
import pandas as pd
import numpy as np
import scipy.io as scio
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
import pytorch_lightning as pl

from . import DATA_TYPE_REGISTRY

class DRDataset():
    def __init__(self, dataset_name="Fdataset", drug_neighbor_num=15, disease_neighbor_num=15):
        assert dataset_name in ["Cdataset", "Fdataset", "DNdataset", "lrssl", "hdvd"]
        self.dataset_name = dataset_name
        if dataset_name=="lrssl":
            old_data = load_DRIMC(name=dataset_name)
        elif dataset_name=="hdvd":
            old_data = load_HDVD()
        else:
            old_data = scio.loadmat(f"dataset/{dataset_name}.mat")

        self.drug_sim = old_data["drug"].astype(np.float)
        self.disease_sim = old_data["disease"].astype(np.float)
        self.drug_name = old_data["Wrname"].reshape(-1)
        self.drug_num = len(self.drug_name)
        self.disease_name = old_data["Wdname"].reshape(-1)
        self.disease_num = len(self.disease_name)
        self.interactions = old_data["didr"].T

        self.drug_edge = self.build_graph(self.drug_sim, drug_neighbor_num)
        self.disease_edge = self.build_graph(self.disease_sim, disease_neighbor_num)
        pos_num = self.interactions.sum()
        neg_num = np.prod(self.interactions.shape) - pos_num
        self.pos_weight = neg_num / pos_num
        print(f"dataset:{dataset_name}, drug:{self.drug_num}, disease:{self.disease_num}, pos weight:{self.pos_weight}")

    def build_graph(self, sim, num_neighbor):
        if num_neighbor>sim.shape[0] or num_neighbor<0:
            num_neighbor = sim.shape[0]
        neighbor = np.argpartition(-sim, kth=num_neighbor, axis=1)[:, :num_neighbor]
        row_index = np.arange(neighbor.shape[0]).repeat(neighbor.shape[1])
        col_index = neighbor.reshape(-1)
        edge_index = torch.from_numpy(np.array([row_index, col_index]).astype(int))
        values = torch.ones(edge_index.shape[1])
        values = torch.from_numpy(sim[row_index, col_index]).float()*values
        return (edge_index, values, sim.shape)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("dataset config")
        parser.add_argument("--dataset_name", default="Fdataset",
                                   choices=["Cdataset", "Fdataset", "lrssl", "hdvd"])
        parser.add_argument("--drug_neighbor_num", default=25, type=int)
        parser.add_argument("--disease_neighbor_num", default=25, type=int)
        return parent_parser



class Dataset():
    def __init__(self, dataset, mask, fill_unkown=True, stage="train", **kwargs):
        mask = mask.astype(bool)
        self.stage = stage
        self.one_mask = torch.from_numpy(dataset.interactions>0)
        row, col = np.nonzero(mask&dataset.interactions.astype(bool))
        self.valid_row = torch.tensor(np.unique(row))
        self.valid_col = torch.tensor(np.unique(col))
        if not fill_unkown:
            row_idx, col_idx = np.nonzero(mask)
            self.interaction_edge = torch.LongTensor([row_idx, col_idx]).contiguous()
            self.label = torch.from_numpy(dataset.interactions[mask]).float().contiguous()
            self.valid_mask = torch.ones_like(self.label, dtype=torch.bool)
            self.matrix_mask = torch.from_numpy(mask)
        else:
            row_idx, col_idx = torch.meshgrid(torch.arange(mask.shape[0]), torch.arange(mask.shape[1]))
            self.interaction_edge = torch.stack([row_idx.reshape(-1), col_idx.reshape(-1)])
            self.label = torch.clone(torch.from_numpy(dataset.interactions)).float()
            self.label[~mask] = 0
            self.valid_mask = torch.from_numpy(mask)
            self.matrix_mask = torch.from_numpy(mask)

        self.drug_edge = dataset.drug_edge
        self.disease_edge = dataset.disease_edge

        self.u_embedding = torch.from_numpy(dataset.drug_sim).float()
        self.v_embedding = torch.from_numpy(dataset.disease_sim).float()

        self.mask = torch.from_numpy(mask)
        pos_num = self.label.sum().item()
        neg_num = np.prod(self.mask.shape) - pos_num
        self.pos_weight = neg_num / pos_num

    def __str__(self):
        return f"{self.__class__.__name__}(shape={self.mask.shape}, interaction_num={len(self.interaction_edge)}, pos_weight={self.pos_weight})"

    @property
    def size_u(self):
        return self.mask.shape[0]

    @property
    def size_v(self):
        return self.mask.shape[1]

    def get_u_edge(self, union_graph=False):
        edge_index, value, size = self.drug_edge
        if union_graph:
            size = (self.size_u+self.size_v, )*2
        return edge_index, value, size

    def get_v_edge(self, union_graph=False):
        edge_index, value, size = self.disease_edge
        if union_graph:
            edge_index = edge_index + torch.tensor(np.array([[self.size_u], [self.size_u]]))
            size = (self.size_u + self.size_v,) * 2
        return edge_index, value, size

    def get_uv_edge(self, union_graph=False):
        train_mask = self.mask if self.stage=="train" else ~self.mask
        train_one_mask = train_mask & self.one_mask
        edge_index = torch.nonzero(train_one_mask).T
        value = torch.ones(edge_index.shape[1])
        size =  (self.size_u, self.size_v)
        if union_graph:
            edge_index = edge_index + torch.tensor([[0], [self.size_u]])
            size = (self.size_u + self.size_v,) * 2
        return edge_index, value, size

    def get_vu_edge(self, union_graph=False):
        edge_index, value, size = self.get_uv_edge(union_graph=union_graph)
        edge_index = reversed(edge_index)
        return edge_index, value, size

    def get_union_edge(self, union_type="u-uv-vu-v"):
        types = union_type.split("-")
        edges = []
        size = (self.size_u+self.size_v, )*2
        for type in types:
            assert type in ["u","v","uv","vu"]
            edge = self.__getattribute__(f"get_{type}_edge")(union_graph=True)
            edges.append(edge)
        edge_index = torch.cat([edge[0] for edge in edges], dim=1)
        value = torch.cat([edge[1] for edge in edges], dim=0)
        return edge_index, value, size

    @staticmethod
    def collate_fn(batch):
        return batch

