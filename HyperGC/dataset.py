from typing import Optional
import os.path as osp
import pickle
import os

import torch
from torch_scatter import scatter_add


class BaseDataset(object):
    def __init__(self, name: str, device: str = 'cpu'):
        self.name = name
        self.device = device

        self.load_dataset()
        self.preprocess_dataset()

    def load_dataset(self):
        self.features = torch.load('data/{0}/X.pt'.format(self.name))
        self.hyperedge_index = torch.load('data/{0}/H.pt'.format(self.name))
        self.labels = torch.load('data/{0}/Y.pt'.format(self.name))

    def load_splits(self, seed: int):
        
        path=f'data/{self.name}/edge_bucket.pickle'
        with open(path, "rb") as f : 
            split = pickle.load(f)
        data=split[seed]
        return data

        
        
    def preprocess_dataset(self):
        self.hypergraph={}
        for i in range(len(self.hyperedge_index[1])):
            if self.hyperedge_index[1][i].item() in self.hypergraph:
                self.hypergraph[self.hyperedge_index[1][i].item()].append(self.hyperedge_index[0][i].item())
            
            else:
                self.hypergraph[self.hyperedge_index[1][i].item()]=[self.hyperedge_index[0][i].item()]
        
        self.num_nodes = int(self.hyperedge_index[0].max()) + 1
        self.num_edges = int(self.hyperedge_index[1].max()) + 1
        self.n_class = int(torch.unique(self.labels).shape[0])
        weight = torch.ones(self.num_edges)
        Dn = scatter_add(weight[self.hyperedge_index[1]], self.hyperedge_index[0], dim=0, dim_size=self.num_nodes)
        De = scatter_add(torch.ones(self.hyperedge_index.shape[1]), self.hyperedge_index[1], dim=0, dim_size=self.num_edges)

        self.to(self.device)

    def to(self, device: str):
        self.features = self.features.to(device)
        self.hyperedge_index = self.hyperedge_index.to(device)
        self.labels = self.labels.to(device)
        self.device = device
        return self



class CiteseercitationDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('citeseer_cite', **kwargs)

class CoracitationDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cora_cite', **kwargs)

class PubmedcitationDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('pubmed_cite', **kwargs)


class CoraCoauthorshipDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cora_coauth', **kwargs)


class DBLPCopublicationDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('dblp_copub', **kwargs)

class DBLPCoauthorshipDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('dblp_coauth', **kwargs)

class AminerDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('aminer', **kwargs)

class IMDBDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('imdb', **kwargs)

class ModelNet40Dataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('modelnet_40', **kwargs)

class NewsDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('news', **kwargs)

class HouseDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('house', **kwargs)

