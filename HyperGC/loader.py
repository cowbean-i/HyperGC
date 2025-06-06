from .dataset import (
    CiteseercitationDataset,
    CoracitationDataset,
    PubmedcitationDataset,
    CoraCoauthorshipDataset,
    DBLPCopublicationDataset,
    DBLPCoauthorshipDataset,
    AminerDataset,
    IMDBDataset,
    ModelNet40Dataset,
    NewsDataset,
    HouseDataset
    )


class DatasetLoader(object):
    def __init__(self):
        pass

    def load(self, dataset_name: str = 'cora_cite'):                    
        if dataset_name == 'citeseer_cite':
            return CiteseercitationDataset()
        elif dataset_name == 'cora_cite':
            return CoracitationDataset()        
        elif dataset_name == 'pubmed_cite':
            return PubmedcitationDataset()       
        elif dataset_name == 'cora_coauth':
            return CoraCoauthorshipDataset()
        elif dataset_name == 'dblp_copub':
            return DBLPCopublicationDataset()
        elif dataset_name == 'dblp_coauth':
            return DBLPCoauthorshipDataset()
        elif dataset_name == 'aminer':
            return AminerDataset()        
        elif dataset_name == 'imdb':
            return IMDBDataset()        
        elif dataset_name == 'modelnet_40':
            return ModelNet40Dataset()        
        elif dataset_name == 'news':
            return NewsDataset()
        elif dataset_name == 'house':        
            return HouseDataset()
        else:
            assert False

"""
'citeseer_cite' 'cora_cite' 'pubmed_cite' 'cora_coauth' 'cora_coauth' 'dblp_copub' 'dblp_coauth' 'aminer' 'imdb' 'modelnet_40' 'news' 'house'
"""