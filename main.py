from functools import partial
from itertools import product

import re
import sys
import argparse
from datasets import get_dataset
from train_eval import single_train_test, cross_validation_with_label
from res_gcn import ResGCN
import random
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

DATA_SOCIAL = ['COLLAB', 'IMDB-BINARY', 'IMDB-MULTI']
DATA_SOCIAL += ['REDDIT-MULTI-5K', 'REDDIT-MULTI-12K', 'REDDIT-BINARY']
DATA_BIO = ['MUTAG', 'NCI1', 'PROTEINS', 'DD', 'ENZYMES', 'PTC_MR']
DATA_REDDIT = [
    data for data in DATA_BIO + DATA_SOCIAL if "REDDIT" in data]
DATA_NOREDDIT = [
    data for data in DATA_BIO + DATA_SOCIAL if "REDDIT" not in data]
DATA_SUBSET_STUDY = ['COLLAB', 'IMDB-BINARY', 'IMDB-MULTI',
                     'NCI1', 'PROTEINS', 'DD']
DATA_SUBSET_STUDY_SUP = [
    d for d in DATA_SOCIAL + DATA_BIO if d not in DATA_SUBSET_STUDY]
DATA_SUBSET_FAST = ['IMDB-BINARY', 'PROTEINS', 'IMDB-MULTI', 'ENZYMES']
DATA_IMAGES = ['MNIST', 'MNIST_SUPERPIXEL', 'CIFAR10']


str2bool = lambda x: x.lower() == "true"
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default="benchmark")
parser.add_argument('--data_root', type=str, default="datasets")
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=500)
parser.add_argument('--epoch_select', type=str, default='test_max')
parser.add_argument('--n_layers_feat', type=int, default=1)
parser.add_argument('--n_layers_conv', type=int, default=3)
parser.add_argument('--n_layers_fc', type=int, default=2)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--global_pool', type=str, default="sum")
parser.add_argument('--skip_connection', type=str2bool, default=False)
parser.add_argument('--res_branch', type=str, default="BNConvReLU")
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--edge_norm', type=str2bool, default=True)
parser.add_argument('--with_eval_mode', type=str2bool, default=True)
parser.add_argument('--dataset', type=str, default="MUTAG")
parser.add_argument('--suffix', type=int, default=0)
parser.add_argument('--n_percents', type=float, default=0.3)
parser.add_argument('--eta', type=float, default=1.0)
parser.add_argument('--num_train', type=int, default=50)
parser.add_argument('--num_val', type=int, default=50)
parser.add_argument('--imb_ratio', type=float, default=0.1)
parser.add_argument('--threshold', type=float, default=0.9)
parser.add_argument('--tau', type=float, default=0.5)
args = parser.parse_args()


def create_n_filter_triples(datasets, feat_strs, nets, gfn_add_ak3=False,
                            gfn_reall=True, reddit_odeg10=False,
                            dd_odeg10_ak1=False):
    triples = [(d, f, n) for d, f, n in product(datasets, feat_strs, nets)]
    triples_filtered = []
    for dataset, feat_str, net in triples:
        # Add ak3 for GFN.
        if gfn_add_ak3 and 'GFN' in net:
            feat_str += '+ak3'
        # Remove edges for GFN.
        if gfn_reall and 'GFN' in net:
            feat_str += '+reall'
        # Replace degree feats for REDDIT datasets (less redundancy, faster).
        if reddit_odeg10 and dataset in [
                'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']:
            feat_str = feat_str.replace('odeg100', 'odeg10')
        # Replace degree and akx feats for dd (less redundancy, faster).
        if dd_odeg10_ak1 and dataset in ['DD']:
            feat_str = feat_str.replace('odeg100', 'odeg10')
            feat_str = feat_str.replace('ak3', 'ak1')
        triples_filtered.append((dataset, feat_str, net))
    return triples_filtered


def get_model_with_default_configs(model_name,
                                   num_feat_layers=args.n_layers_feat,
                                   num_conv_layers=args.n_layers_conv,
                                   num_fc_layers=args.n_layers_fc,
                                   residual=args.skip_connection,
                                   hidden=args.hidden):
    # More default settings.
    res_branch = args.res_branch
    global_pool = args.global_pool
    dropout = args.dropout
    edge_norm = args.edge_norm

    # modify default architecture when needed
    if model_name.find('_') > 0:
        num_conv_layers_ = re.findall('_conv(\d+)', model_name)
        if len(num_conv_layers_) == 1:
            num_conv_layers = int(num_conv_layers_[0])
            print('[INFO] num_conv_layers set to {} as in {}'.format(
                num_conv_layers, model_name))
        num_fc_layers_ = re.findall('_fc(\d+)', model_name)
        if len(num_fc_layers_) == 1:
            num_fc_layers = int(num_fc_layers_[0])
            print('[INFO] num_fc_layers set to {} as in {}'.format(
                num_fc_layers, model_name))
        residual_ = re.findall('_res(\d+)', model_name)
        if len(residual_) == 1:
            residual = bool(int(residual_[0]))
            print('[INFO] residual set to {} as in {}'.format(
                residual, model_name))
        gating = re.findall('_gating', model_name)
        if len(gating) == 1:
            global_pool += "_gating"
            print('[INFO] add gating to global_pool {} as in {}'.format(
                global_pool, model_name))
        dropout_ = re.findall('_drop([\.\d]+)', model_name)
        if len(dropout_) == 1:
            dropout = float(dropout_[0])
            print('[INFO] dropout set to {} as in {}'.format(
                dropout, model_name))
        hidden_ = re.findall('_dim(\d+)', model_name)
        if len(hidden_) == 1:
            hidden = int(hidden_[0])
            print('[INFO] hidden set to {} as in {}'.format(
                hidden, model_name))



    if model_name.startswith('ResGFN'):
        collapse = True if 'flat' in model_name else False
        def foo(dataset):
            return ResGCN(dataset, hidden, num_feat_layers, num_conv_layers,
                          num_fc_layers, gfn=True, collapse=collapse,
                          residual=residual, res_branch=res_branch,
                          global_pool=global_pool, dropout=dropout,
                          edge_norm=edge_norm)
    elif model_name.startswith('ResGCN'):
        def foo(dataset):
            return ResGCN(dataset, hidden, num_feat_layers, num_conv_layers,
                          num_fc_layers, gfn=False, collapse=False,
                          residual=residual, res_branch=res_branch,
                          global_pool=global_pool, dropout=dropout,
                          edge_norm=edge_norm)
    else:
        raise ValueError("Unknown model {}".format(model_name))
    return foo


def run_exp_lib(dataset_feat_net_triples,
                get_model=get_model_with_default_configs):
    results = []
    exp_nums = len(dataset_feat_net_triples)
    print("-----\nTotal %d experiments in this run:" % exp_nums)
    for exp_id, (dataset_name, feat_str, net) in enumerate(
            dataset_feat_net_triples):
        print('{}/{} - {} - {} - {}'.format(
            exp_id+1, exp_nums, dataset_name, feat_str, net))
    print("Here we go..")
    sys.stdout.flush()
    for exp_id, (dataset_name, feat_str, net) in enumerate(
            dataset_feat_net_triples):
        print('-----\n{}/{} - {} - {} - {}'.format(
            exp_id+1, exp_nums, dataset_name, feat_str, net))
        sys.stdout.flush()
        dataset = get_dataset(
            dataset_name, sparse=True, feat_str=feat_str, root=args.data_root)
        model_func = get_model(net)
        acc, f1, info = cross_validation_with_label(
                dataset,
                model_func,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                lr_decay_factor=args.lr_decay_factor,
                lr_decay_step_size=args.lr_decay_step_size,
                weight_decay=0,
                epoch_select=args.epoch_select,
                num_train=args.num_train,
                num_val=args.num_val,
                imb_ratio=args.imb_ratio,
                n_percents=args.n_percents,
                with_eval_mode=args.with_eval_mode,
                dataset_name=args.dataset,
                eta=args.eta, 
                threshold=args.threshold, tau=args.tau)
    return acc, f1, info


def run_exp_benchmark():
    # Run GFN, GFN (light), GCN
    print('[INFO] running standard benchmarks..')
    # datasets = DATA_BIO + DATA_SOCIAL
    datasets = [args.dataset]
    feat_strs = ['deg+odeg100']
    # nets = ['ResGFN', 'ResGFN_conv0_fc2', 'ResGCN']
    nets = ['ResGCN']
    acc, f1, info = run_exp_lib(create_n_filter_triples(datasets, feat_strs, nets,
                                        gfn_add_ak3=True,
                                        reddit_odeg10=True,
                                        dd_odeg10_ak1=True))
    return acc, f1, info


if __name__ == '__main__':
    from tqdm import tqdm
    pbar = tqdm(range(10), unit='run')
    seed = 1
    accs, f1s = [], []
    for count in pbar:
        random.seed(seed + count)
        np.random.seed(seed + count)
        torch.manual_seed(seed + count)
        torch.cuda.manual_seed(seed + count)
        torch.cuda.manual_seed_all(seed + count)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        acc, f1, info = run_exp_benchmark()
        torch.save(info, f"logs/{args.dataset}_{count}.log")
        print("Acc:", acc.item(), "F1:", f1.item())
        accs.append(acc)
        f1s.append(f1)
    print('Test Acc: {:.4f} ± {:.4f}, Test F1: {:.4f} ± {:.4f}'.
          format(np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)))
