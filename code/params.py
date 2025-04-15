import argparse
import sys

argv = sys.argv
dataset = argv[1]



def CoraFull_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--batch', type=int, default=20000)  # For large datast
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=20)  # 20
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--num_train', type=int, default=10)
    parser.add_argument('--num_test', type=int, default=100)
    parser.add_argument('--n_way', type=int, default=10)
    parser.add_argument('--k_shot', type=int, default=1)
    parser.add_argument('--m_qry', type=int, default=10)

    ## encoder
    parser.add_argument('--num_head', type=int, default=2)  # For GAT
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--num_layers_gat', type=int, default=1)
    parser.add_argument('--feat_drop', type=float, default=0.7)
    parser.add_argument('--attn_drop', type=float, default=0.7)
    parser.add_argument('--negative_slope', type=float, default=0.1)

    parser.add_argument('--beta', type=float, default=0.1)

    ## graph tree contrast
    parser.add_argument('--tree_height', type=int, default=3)
    parser.add_argument('--init_w', type=float, default=0.7)
    parser.add_argument('--tau', type=float, default=0.3)
    parser.add_argument('--t', type=float, default=1e-6)

    ## MAML cls
    parser.add_argument('--cls_update_lr', type=float, default=0.05)
    parser.add_argument('--cls_update_step', type=int, default=20)

    ## optimizer
    parser.add_argument('--cls_lr', type=float, default=0.005)
    parser.add_argument('--cls_weight_decay', type=float, default=1e-4)
    parser.add_argument('--encoder_lr', type=float, default=0.005)
    parser.add_argument('--encoder_weight_decay', type=float, default=5e-4)

    ## Log--dl
    parser.add_argument('--solver', type=str, default='newton-cg')
    parser.add_argument('--iter_clf', type=int, default=1000)
    #####################################

    args, _ = parser.parse_known_args()
    return args


def Clothing_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--batch', type=int, default=20000)  # For large datast
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=4701)
    parser.add_argument('--patience', type=int, default=20)  # 20
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--num_train', type=int, default=5)
    parser.add_argument('--num_test', type=int, default=100)
    parser.add_argument('--n_way', type=int, default=10)
    parser.add_argument('--k_shot', type=int, default=1)
    parser.add_argument('--m_qry', type=int, default=10)

    ## encoder
    parser.add_argument('--num_head', type=int, default=2)  # For GAT
    parser.add_argument('--num_hidden', type=int, default=128)
    parser.add_argument('--num_layers_gat', type=int, default=1)
    parser.add_argument('--feat_drop', type=float, default=0.2)
    parser.add_argument('--attn_drop', type=float, default=0.5)
    parser.add_argument('--negative_slope', type=float, default=0.4)

    parser.add_argument('--beta', type=float, default=0.001)

    ## graph tree contrast
    parser.add_argument('--tree_height', type=int, default=3)
    parser.add_argument('--init_w', type=float, default=0.7)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--t', type=float, default=1e-6)

    ## MAML cls
    parser.add_argument('--cls_update_lr', type=float, default=0.3)
    parser.add_argument('--cls_update_step', type=int, default=20)

    ## optimizer
    parser.add_argument('--cls_lr', type=float, default=0.01)
    parser.add_argument('--cls_weight_decay', type=float, default=1e-4)
    parser.add_argument('--encoder_lr', type=float, default=0.005)
    parser.add_argument('--encoder_weight_decay', type=float, default=5e-4)

    ## Log--dl
    parser.add_argument('--solver', type=str, default='newton-cg')
    parser.add_argument('--iter_clf', type=int, default=700)
    #####################################

    args, _ = parser.parse_known_args()
    return args


def dblp_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--batch', type=int, default=40672)  # For large datast
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1214)
    parser.add_argument('--patience', type=int, default=20)  # 20
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--num_train', type=int, default=10)
    parser.add_argument('--num_test', type=int, default=100)
    parser.add_argument('--n_way', type=int, default=10)
    parser.add_argument('--k_shot', type=int, default=1)
    parser.add_argument('--m_qry', type=int, default=10)

    ## encoder
    parser.add_argument('--num_head', type=int, default=2)  # For GAT
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--num_layers_gat', type=int, default=1)
    parser.add_argument('--feat_drop', type=float, default=0.3)
    parser.add_argument('--attn_drop', type=float, default=0.4)
    parser.add_argument('--negative_slope', type=float, default=0.4)

    parser.add_argument('--beta', type=float, default=0.001)

    ## graph tree contrast
    parser.add_argument('--tree_height', type=int, default=3)
    parser.add_argument('--init_w', type=float, default=0.6)
    parser.add_argument('--tau', type=float, default=0.3)
    parser.add_argument('--t', type=float, default=1e-5)

    ## MAML cls
    parser.add_argument('--cls_update_lr', type=float, default=0.1)
    parser.add_argument('--cls_update_step', type=int, default=20)

    ## optimizer
    parser.add_argument('--cls_lr', type=float, default=0.005)
    parser.add_argument('--cls_weight_decay', type=float, default=1e-4)
    parser.add_argument('--encoder_lr', type=float, default=0.005)
    parser.add_argument('--encoder_weight_decay', type=float, default=5e-4)

    ## Log--dl
    parser.add_argument('--solver', type=str, default='newton-cg')
    parser.add_argument('--iter_clf', type=int, default=700)
    #####################################

    args, _ = parser.parse_known_args()
    return args


def arxiv_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset_name', type=str, default=dataset)
    parser.add_argument('--batch', type=int, default=20000)  # For large datast
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=20)  # 20
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--num_train', type=int, default=10)
    parser.add_argument('--num_test', type=int, default=100)
    parser.add_argument('--n_way', type=int, default=10)
    parser.add_argument('--k_shot', type=int, default=1)
    parser.add_argument('--m_qry', type=int, default=10)

    ## encoder
    parser.add_argument('--num_head', type=int, default=2)  # For GAT
    parser.add_argument('--num_hidden', type=int, default=128)
    parser.add_argument('--num_layers_gat', type=int, default=1)
    parser.add_argument('--feat_drop', type=float, default=0.2)
    parser.add_argument('--attn_drop', type=float, default=0.4)
    parser.add_argument('--negative_slope', type=float, default=0.4)

    parser.add_argument('--beta', type=float, default=0.01)

    ## graph tree contrast
    parser.add_argument('--tree_height', type=int, default=3)
    parser.add_argument('--init_w', type=float, default=0.6)
    parser.add_argument('--tau', type=float, default=0.2)
    parser.add_argument('--t', type=float, default=1e-6)

    ## MAML cls
    parser.add_argument('--cls_update_lr', type=float, default=0.3)
    parser.add_argument('--cls_update_step', type=int, default=15)

    ## optimizer
    parser.add_argument('--cls_lr', type=float, default=0.005)
    parser.add_argument('--cls_weight_decay', type=float, default=1e-4)
    parser.add_argument('--encoder_lr', type=float, default=0.01)
    parser.add_argument('--encoder_weight_decay', type=float, default=1e-4)

    ## Log--dl
    parser.add_argument('--solver', type=str, default='newton-cg')
    parser.add_argument('--iter_clf', type=int, default=600)
    #####################################

    args, _ = parser.parse_known_args()
    return args


def set_params():
    if dataset == "CoraFull":
        args = CoraFull_params()
        args.pyg = False
        args.big = False
    if dataset == "Amazon-Clothing":
        args = Clothing_params()
        args.pyg = False
        args.big = False
    if dataset == "ogbn-arxiv":
        args = arxiv_params()
        args.pyg = False
        args.big = True
    if dataset == "dblp":
        args = dblp_params()
        args.pyg = False
        args.big = True
    return args
