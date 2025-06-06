from typing import Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_add, scatter
from torch.nn import Parameter
from HyperGC.layers import ProposedConv
from itertools import zip_longest,combinations


class HyperEncoder(nn.Module):
    def __init__(self, in_dim, edge_dim, node_dim, num_layers=2, act: Callable = nn.PReLU()):
        super(HyperEncoder, self).__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.act = act

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(ProposedConv(self.in_dim, self.edge_dim, self.node_dim, cached=False, act=act))
        else:
            self.convs.append(ProposedConv(self.in_dim, self.edge_dim, self.node_dim, cached=False, act=act))
            for _ in range(self.num_layers - 2):
                self.convs.append(ProposedConv(self.node_dim, self.edge_dim, self.node_dim, cached=False, act=act))
            self.convs.append(ProposedConv(self.node_dim, self.edge_dim, self.node_dim, cached=False, act=act))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, hyperedge_index: Tensor, num_nodes: int, num_edges: int):
        for i in range(self.num_layers):
            x, e = self.convs[i](x, hyperedge_index, num_nodes, num_edges)
            x = self.act(x)
        return x, e # act, act





class Decoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """

    def __init__(self, hidden_dim, ncell, lstm_layers):
        """
        Initiate Decoder

        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """

        super(Decoder, self).__init__()
        
        self.dec_hid_dim = hidden_dim
        self.ncell=ncell
        self.lstm_layers=lstm_layers
        self.variablecell=True

        self.input_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        # Used for propagating .cuda() command
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)


        self.input_dim = hidden_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(hidden_dim, hidden_dim)
        self.context_linear = nn.Linear(hidden_dim, hidden_dim)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.reset_parameters()



    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


    def reset_parameters(self):
        self.input_to_hidden.reset_parameters()
        self.hidden_to_hidden.reset_parameters()
        self.hidden_out.reset_parameters()
        self.input_linear.reset_parameters()
        self.context_linear.reset_parameters()

    def forward(self, n,decoder_input,hidden,masked_nodes,hopmask,edge_size,top_p):
        
        num_genhe=len(decoder_input)
        ncell = 2
        edge_size=torch.tensor(edge_size,device=n.device)
        


        mask = self.mask.repeat(n.size(0)).unsqueeze(0).repeat(num_genhe, 1)
        self.init_inf(mask.size()) 

        
        runner = self.runner.repeat(n.size(0))
        for i in range(n.size(0)):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(num_genhe, -1).long()

        outputs = []
        pointers = []
        seq_score = []
     
        for i, nodes in enumerate(masked_nodes):
            mask[i, nodes] = 0
        

        for i in range(ncell-1): 
            
            num_genhe=decoder_input.size(0)
            h,c=hidden
            gates = self.input_to_hidden(decoder_input) + self.hidden_to_hidden(h)
            input, forget, cell, out = gates.chunk(4, 1)

            input = torch.sigmoid(input) #i_t
            forget = torch.sigmoid(forget) #f_t
            cell = torch.tanh(cell) #tilda c_t
            out = torch.sigmoid(out) 
            c = (forget * c) + ( input * cell)

            hi = out * torch.tanh(c)

            attention_mask=torch.eq(mask, 0)[:num_genhe]
            key=self.context_linear(n)
            query=self.input_linear(hi)
            att = torch.matmul(query,key.transpose(-2,-1)) 
            if len(att[attention_mask]) > 0:
                att[attention_mask] = self.inf[attention_mask]
            outs = self.softmax(att)
            hidden_t = torch.mm(outs,key)
            h = torch.tanh(self.hidden_out(torch.cat((hidden_t, hi), 1)))
            
            masked_outs = outs * mask[:num_genhe]

            sorted_probs, sorted_indices = torch.sort(masked_outs, descending=True, dim=-1)

            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            topp_mask = cumulative_probs <= top_p

            topp_mask[:, 1:] = topp_mask[:, :-1].clone()
            topp_mask[:, 0] = True  
            masked_probs = sorted_probs * topp_mask.float()
        
            masked_probs /= masked_probs.sum(dim=-1, keepdim=True)

            sampled_indices = torch.multinomial(masked_probs, num_samples=1)

            indices= sorted_indices.gather(1, sampled_indices).squeeze()
            if indices.dim()==0:
                indices=indices.unsqueeze(0)
                one_hot_pointers = (runner[:num_genhe] == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()
            else:
                one_hot_pointers = (runner[:num_genhe] == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()
            mask[:num_genhe]  = mask[:num_genhe] * (1 - one_hot_pointers)
            decoder_input = n[indices]

            outputs.append(outs)
            pointers.append(indices.cpu().detach())
            
        

        return (outputs, pointers), hidden

class GenCL(nn.Module):
    def __init__(self, encoder: HyperEncoder, decoder: Decoder, proj_dim: int, x_dim):
        super(GenCL, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        

        self.node_dim = self.encoder.node_dim
        self.edge_dim = self.encoder.edge_dim

        self.dec_hid_dim = self.decoder.dec_hid_dim
        self.ncell = self.decoder.ncell
        self.lstm_layers = self.decoder.lstm_layers

        self.fc1_n = nn.Linear(self.node_dim, proj_dim)
        self.fc2_n = nn.Linear(proj_dim, self.node_dim)
        self.fc1_e = nn.Linear(self.edge_dim, proj_dim)
        self.fc2_e = nn.Linear(proj_dim, self.edge_dim)
        self.fc_nodes2edge = nn.Linear(self.edge_dim,self.edge_dim)

        self.h0 = Parameter(torch.FloatTensor(1), requires_grad=True)
        self.c0 = Parameter(torch.FloatTensor(1), requires_grad=True)


        self.disc = nn.Bilinear(self.node_dim, self.edge_dim, 1)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        self.fc1_n.reset_parameters()
        self.fc2_n.reset_parameters()
        self.fc1_e.reset_parameters()
        self.fc2_e.reset_parameters()
        self.disc.reset_parameters()
        
    def forward(self, x: Tensor, hyperedge_index: Tensor,
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None):
        if num_nodes is None:
            num_nodes = int(hyperedge_index[0].max()) + 1
        if num_edges is None:
            num_edges = int(hyperedge_index[1].max()) + 1

        node_idx = torch.arange(0, num_nodes, device=x.device)
        edge_idx = torch.arange(num_edges, num_edges + num_nodes, device=x.device)
        self_loop = torch.stack([node_idx, edge_idx])
        self_loop_hyperedge_index = torch.cat([hyperedge_index, self_loop], 1)
        n, e = self.encoder(x, self_loop_hyperedge_index, num_nodes, num_edges + num_nodes)
        return n, e[:num_edges] 

    def without_selfloop(self, x: Tensor, hyperedge_index: Tensor, node_mask: Optional[Tensor] = None,
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None):
        if num_nodes is None:
            num_nodes = int(hyperedge_index[0].max()) + 1
        if num_edges is None:
            num_edges = int(hyperedge_index[1].max()) + 1

        if node_mask is not None:
            node_idx = torch.where(~node_mask)[0]
            edge_idx = torch.arange(num_edges, num_edges + len(node_idx), device=x.device)
            self_loop = torch.stack([node_idx, edge_idx])
            self_loop_hyperedge_index = torch.cat([hyperedge_index, self_loop], 1)
            n, e = self.encoder(x, self_loop_hyperedge_index, num_nodes, num_edges + len(node_idx))
            return n, e[:num_edges]
        else:
            return self.encoder(x, hyperedge_index, num_nodes, num_edges)

    def f(self, x, tau):
        return torch.exp(x / tau)
    def node_projection(self, z: Tensor):
        return self.fc2_n(F.elu(self.fc1_n(z)))

    def edge_projection(self, z: Tensor):
        return self.fc2_e(F.elu(self.fc1_e(z)))
    
    def cosine_similarity(self, z1: Tensor, z2: Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def disc_similarity(self, z1: Tensor, z2: Tensor):
        return torch.sigmoid(self.disc(z1, z2)).squeeze()


    
    def generator_partial(self, n: Tensor ,e: Tensor, queries,masked_nodes, edge_size, top_p):

        masked_nodes=[x[:-1] for x in masked_nodes]
        num_nodes=n.size(0)
        hopmask=np.zeros((num_nodes,num_nodes),dtype=bool)
        num_genhe=len(queries)

        h0=torch.randn((num_genhe,self.dec_hid_dim),device=n.device)#self.h0.unsqueeze(0).repeat(num_genhe,self.dec_hid_dim) #n[s]
        c0=torch.randn((num_genhe,self.dec_hid_dim),device=n.device)#self.c0.unsqueeze(0).repeat(num_genhe,self.dec_hid_dim) #totale[drop_idx]
#c0=torch.randn((num_genhe,self.dec_hid_dim),device=n.device)
        decoder_hidden0=(h0,c0)
        decoder_input0=queries
        (outputs, pointers), decoder_hidden = self.decoder(n,decoder_input0,decoder_hidden0,masked_nodes, hopmask,edge_size,top_p)

        v=[]
        e=[]

        num_idx=-1
        for i in range(len(masked_nodes)):
            num_idx+=1
            v+=masked_nodes[i]+[pointers[0][i].item()]
            for _ in range(len(masked_nodes[i])+1):
                e.append(num_idx)

        z=scatter(src=n[v,:], index=torch.tensor(e,device=n.device), dim=0 ,reduce='sum')
        z=self.fc_nodes2edge(z)

        return outputs,pointers,z

    def __semi_loss(self, h1: Tensor, h2: Tensor, tau: float, num_negs: Optional[int]):
        if num_negs is None:
            between_sim = self.f(self.cosine_similarity(h1, h2), tau)
            return -torch.log(between_sim.diag() / between_sim.sum(1)) 

        else:
            pos_sim = self.f(F.cosine_similarity(h1, h2), tau)
            negs = []
            for _ in range(num_negs):
                negs.append(h2[torch.randperm(h2.size(0))])
            negs = torch.stack(negs, dim=-1)
            neg_sim = self.f(F.cosine_similarity(h1.unsqueeze(-1).tile(num_negs), negs), tau)
            return -torch.log(pos_sim / (pos_sim + neg_sim.sum(1)))
        
    def __semi_loss_batch(self, h1: Tensor, h2: Tensor, tau: float, batch_size: int):
        device = h1.device
        num_samples = h1.size(0)
        num_batches = (num_samples - 1) // batch_size + 1
        indices = torch.arange(0, num_samples, device=device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size: (i + 1) * batch_size]
            between_sim = self.f(self.cosine_similarity(h1[mask], h2), tau)

            loss = -torch.log(between_sim[:, i * batch_size: (i + 1) * batch_size].diag() / between_sim.sum(1))
            losses.append(loss)
        return torch.cat(losses)

    def __loss(self, z1: Tensor, z2: Tensor, tau: float, batch_size: Optional[int], 
               num_negs: Optional[int], mean: bool):
        if batch_size is None or num_negs is not None:
            l1 = self.__semi_loss(z1, z2, tau, num_negs)
            l2 = self.__semi_loss(z2, z1, tau, num_negs)
        else:
            l1 = self.__semi_loss_batch(z1, z2, tau, batch_size)
            l2 = self.__semi_loss_batch(z2, z1, tau, batch_size)

        loss = (l1 + l2) * 0.5
        loss = loss.mean() if mean else loss.sum()
        return loss
    def cl_loss_n(self, n1: Tensor, n2: Tensor, node_tau: float, 
                       batch_size: Optional[int] = None, num_negs: Optional[int] = None, 
                       mean: bool = True):
        loss = self.__loss(n1, n2, node_tau, batch_size, num_negs, mean)
        return loss

    def cl_loss_g(self, e1: Tensor, e2: Tensor, edge_tau: float, 
                       batch_size: Optional[int] = None, num_negs: Optional[int] = None, 
                       mean: bool = True):
        loss = self.__loss(e1, e2, edge_tau, batch_size, num_negs, mean)
        return loss
    def cl_loss_m(self, n: Tensor, e: Tensor, hyperedge_index: Tensor, tau: float, 
                              batch_size: Optional[int] = None, mean: bool = True):
        e_perm = e[torch.randperm(e.size(0))]
        n_perm = n[torch.randperm(n.size(0))]
        if batch_size is None:
            pos = self.f(self.disc_similarity(n[hyperedge_index[0]], e[hyperedge_index[1]]), tau)
            neg_n = self.f(self.disc_similarity(n[hyperedge_index[0]], e_perm[hyperedge_index[1]]), tau)
            neg_e = self.f(self.disc_similarity(n_perm[hyperedge_index[0]], e[hyperedge_index[1]]), tau)

            loss_n = -torch.log(pos / (pos + neg_n))
            loss_e = -torch.log(pos / (pos + neg_e))
        else:
            num_samples = hyperedge_index.shape[1]
            num_batches = (num_samples - 1) // batch_size + 1
            indices = torch.arange(0, num_samples, device=n.device)
            
            aggr_pos = []
            aggr_neg_n = []
            aggr_neg_e = []
            for i in range(num_batches):
                mask = indices[i * batch_size: (i + 1) * batch_size]

                pos = self.f(self.disc_similarity(n[hyperedge_index[:, mask][0]], e[hyperedge_index[:, mask][1]]), tau)
                neg_n = self.f(self.disc_similarity(n[hyperedge_index[:, mask][0]], e_perm[hyperedge_index[:, mask][1]]), tau)
                neg_e = self.f(self.disc_similarity(n_perm[hyperedge_index[:, mask][0]], e[hyperedge_index[:, mask][1]]), tau)

                if pos.dim()==0:
                    pos=pos.unsqueeze(0)
                if neg_n.dim()==0:
                    neg_n=neg_n.unsqueeze(0)
                if neg_e.dim()==0:
                    neg_e=neg_e.unsqueeze(0)
                    
                aggr_pos.append(pos)
                aggr_neg_n.append(neg_n)
                aggr_neg_e.append(neg_e)

                
            aggr_pos = torch.concat(aggr_pos)
            aggr_neg_n = torch.concat(aggr_neg_n)
            aggr_neg_e = torch.concat(aggr_neg_e)

            loss_n = -torch.log(aggr_pos / (aggr_pos + aggr_neg_n))
            loss_e = -torch.log(aggr_pos / (aggr_pos + aggr_neg_e))

        loss_n = loss_n[~torch.isnan(loss_n)]
        loss_e = loss_e[~torch.isnan(loss_e)]
        loss = loss_n + loss_e
        loss = loss.mean() if mean else loss.sum()
        return loss 
    
    def ptr_loss_ce(self, prob, masked_nodes:list, gen_node):
        loss=0
        prob=prob[0]
        gen_node=gen_node[0].tolist()
        for idx in range(len(masked_nodes)):
            target_sample=masked_nodes[idx][-1]
            predict_sample=gen_node
            prob_sample=prob[idx]
            sample_loss=0
            
            p=torch.zeros(len(prob[idx]),dtype=torch.float32,device=prob[idx].device)
            p[target_sample]=1

            q=prob_sample
            sample_loss = -torch.sum(p * torch.log(q + 1e-10) + (1 - p) * torch.log(1 - q + 1e-10))
        
                              
            loss+=sample_loss               

        return torch.log(loss+1e-10)
    