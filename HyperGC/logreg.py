import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch_scatter import scatter_add, scatter

class MLP(nn.Module) : 
    
    def __init__(self, in_dim, n_class) :
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(in_dim, n_class)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.linear1.reset_parameters()
        
    def forward(self, x) : 
        x = self.linear1(x)
        return x
        
class MLP_HENN(nn.Module) :
    
    def __init__(self, in_dim, hidden_dim, p = 0.5) : 
        super(MLP_HENN, self).__init__() 
        
        self.classifier1 = nn.Linear(in_dim, hidden_dim)
        self.classifier2 = nn.Linear(hidden_dim, 1)
        self.dropouts = nn.Dropout(p = p)
        self.reset_parameters()

    def reset_parameters(self):
        self.classifier1.reset_parameters()
        self.classifier2.reset_parameters()
        
    def forward(self, x, target_nodes, target_ids: list) : 

        Z = scatter(src = x[target_nodes, :], index = target_ids, dim = 0, reduce = 'sum')
        Z = (self.classifier1(Z)) # No need of Logits
        Z = torch.relu(Z)
        Z = self.dropouts(Z)
        Z = (self.classifier2(Z)) # No need of Logits
        
        return torch.sigmoid(Z).squeeze(-1) # Edge Prediction Probability
