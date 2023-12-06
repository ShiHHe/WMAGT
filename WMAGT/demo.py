from main import parse, train

if __name__=="__main__":
    args = parse(print_help=True)
    # args.n_splits = 10
    # args.dataset_name = "Fdataset"
    args.dataset_name = "Cdataset"
    # args.lr = 1e-3
    # args.lr = 5e-4
    # # args.use_bn = True
    # args.dropout = 0.4
    # 0 -1(all)
    args.disease_neighbor_num = 5
    args.drug_neighbor_num = 5
    # args.disease_feature_topk = 20
    # args.drug_feature_topk = 20
    # args.embedding_dim = 64
    # args.neighbor_embedding_dim = 32
    # args.hidden_dims = (64, 32)
    # args.debug = True
    # args.epochs = 1
    args.train_fill_unknown = False
    # args.comment = "test"
    # args.loss_fn = "focal"

    # args.alpha = 0.5

    train(args, WMAGT)

