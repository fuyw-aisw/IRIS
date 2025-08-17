import os.path as osp
import re

import torch
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from feature_expansion import FeatureExpander
from tu_dataset import TUDatasetExt


def get_dataset(name, sparse=True, feat_str="deg+ak3+reall", root=None, aug=None, aug_ratio=None):
    if root is None or root == '':
        path = osp.join(osp.expanduser('~'), 'pyG_data', name)
    else:
        path = root#osp.join(root, name)
    path = '../' + path
    is_degree = feat_str.find("deg") >= 0
    onehot_maxdeg = re.findall("odeg(\d+)", feat_str)
    onehot_maxdeg = int(onehot_maxdeg[0]) if onehot_maxdeg else None
    k = re.findall("an{0,1}k(\d+)", feat_str)
    k = int(k[0]) if k else 0
    groupd = re.findall("groupd(\d+)", feat_str)
    groupd = int(groupd[0]) if groupd else 0
    remove_edges = re.findall("re(\w+)", feat_str)
    remove_edges = remove_edges[0] if remove_edges else 'none'
    edge_noises_add = re.findall("randa([\d\.]+)", feat_str)
    edge_noises_add = float(edge_noises_add[0]) if edge_noises_add else 0
    edge_noises_delete = re.findall("randd([\d\.]+)", feat_str)
    edge_noises_delete = float(
        edge_noises_delete[0]) if edge_noises_delete else 0
    centrality = feat_str.find("cent") >= 0
    coord = feat_str.find("coord") >= 0

    pre_transform = FeatureExpander(
        degree=is_degree, onehot_maxdeg=onehot_maxdeg, AK=k,
        centrality=centrality, remove_edges=remove_edges,
        edge_noises_add=edge_noises_add, edge_noises_delete=edge_noises_delete,
        group_degree=groupd).transform

    print(aug, aug_ratio)
    dataset = TUDatasetExt(
        path, name, pre_transform=pre_transform,
        use_node_attr=True, processed_filename="data_%s.pt" % feat_str,
        aug=aug, aug_ratio=aug_ratio
    )

    dataset.data.edge_attr = None

    return dataset

def merged_tudatasets(dataset1, dataset2):
    aug1, aug_ratio1 = dataset1.aug, dataset1.aug_ratio
    aug2, aug_ratio2 = dataset2.aug, dataset2.aug_ratio    
    dataset1.aug, dataset1.aug_ratio = "none", None
    pre_transform = dataset1.pre_transform
    
    data_list = [data for data in dataset1] + [data for data in dataset2]
    merged_datasets = TUDatasetExt.__new__(TUDatasetExt)
    
    merged_datasets.root = "" 
    merged_datasets.name = f"merged_{dataset1.name}"
    merged_datasets.use_node_attr = True
    merged_datasets.pre_transform = dataset1.pre_transform
    merged_datasets.processed_filename = "" 
    merged_datasets.aug = "none"
    merged_datasets.aug_ratio = None
    merged_datasets.__indices__ = None
    merged_datasets.transform = dataset1.transform
    
    merged_datasets.data, merged_datasets.slices = merged_datasets.collate(data_list)
    dataset1.aug, dataset1.aug_ratio = aug1, aug_ratio1
    dataset2.aug, dataset2.aug_ratio = aug2, aug_ratio2  
    for attr in ['p_edge_node', 'deg_max', 'deg_min']:
        if hasattr(dataset1, attr):
            setattr(merged_datasets, attr, getattr(dataset1, attr))   
    assert len(merged_datasets) == len(dataset1) + len(dataset2)
    #print(torch.bincount(torch.tensor([data.y.item() for data in merged_datasets])))    
    return merged_datasets
