import argparse
import random
import os
import copy
import yaml
import gc
from tqdm import tqdm
import numpy as np
import torch
import pickle
from HyperGC.loader import DatasetLoader
from HyperGC.models_partial import HyperEncoder,Decoder,GenCL
from HyperGC.utils import drop_features, drop_incidence, drop_hyperedges, valid_node_edge_mask, hyperedge_index_masking, make_hopmask
from torch_scatter import scatter
from HyperGC.evaluation import NC_predictor_linear, HE_predictor

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def train(args):

    if args.task=="node":
        features, hyperedge_index = data.features, data.hyperedge_index
    elif args.task=="edge":
        features, hyperedge_index = data.features, data.edge_split[3].to(args.device)


    num_nodes, num_edges = data.num_nodes, hyperedge_index[1].max().item()+1
    minhop,maxhop=args.minhop,args.maxhop

    if args.hopmask:
        hopmask_file_path=f'./data/{data.name}/distance.npy'
        if os.path.exists(hopmask_file_path):
            distance_matrix=np.load(hopmask_file_path)
        else:
            distance_matrix=make_hopmask(data.hypergraph)
            np.save(hopmask_file_path, distance_matrix)
            
        hopmask = np.where(
        (distance_matrix >= minhop) & (distance_matrix <= maxhop), False, True)
    else:
        hopmask = np.zeros((data.num_nodes,data.num_nodes),dtype=bool)
    hopmask=torch.from_numpy(hopmask).to(args.device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if args.mode=="pr":

        masked_nodes,start,edge_size,drop_idx=drop_hyperedges(hyperedge_index,num_nodes,num_edges,args.drop_hyperedge_rate,args.ncell,args.node_select)

        
        

        n, e = model(data.features, hyperedge_index, num_nodes, num_edges)
        
        nodes=[]
        for x in masked_nodes:
            nodes+=x
        idx=0
        edges=[]
        for x in masked_nodes:
            for i in range(len(x)):
                edges.append(idx)
            idx+=1

        n=model.node_projection(n)
        e=model.edge_projection(e)
        
        queries=scatter(src=n[nodes],index=torch.tensor(edges,device=n.device),dim=0,reduce='sum')
        prob, gen_node,_ = model.generator_partial(n,e,queries,masked_nodes,[2]*len(masked_nodes),top_p=args.top_p)

        loss=model.ptr_loss_ce(prob,masked_nodes,gen_node)
    elif args.mode=="pr+cl":
        masked_nodes1,start1,edge_size1,drop_idx1=drop_hyperedges(hyperedge_index,num_nodes,num_edges,args.drop_hyperedge_rate,args.ncell,args.node_select)
        masked_nodes2,start2,edge_size2,drop_idx2=drop_hyperedges(hyperedge_index,num_nodes,num_edges,args.drop_hyperedge_rate,args.ncell,args.node_select)

        hyperedge_index1 = drop_incidence(hyperedge_index, args.drop_incidence_rate)
        hyperedge_index2 = drop_incidence(hyperedge_index, args.drop_incidence_rate)
        x1 = drop_features(features, args.drop_feature_rate)
        x2 = drop_features(features, args.drop_feature_rate)

        node_mask1, edge_mask1 = valid_node_edge_mask(hyperedge_index1, num_nodes, num_edges)
        node_mask2, edge_mask2 = valid_node_edge_mask(hyperedge_index2, num_nodes, num_edges)
        node_mask = node_mask1 & node_mask2
        edge_mask = edge_mask1 & edge_mask2

        n1, e1 = model(x1, hyperedge_index1, num_nodes, num_edges)
        n2, e2 = model(x2, hyperedge_index2, num_nodes, num_edges)

        n1, n2 = model.node_projection(n1), model.node_projection(n2)
        e1, e2 = model.edge_projection(e1), model.edge_projection(e2)


        loss_n = model.cl_loss_n(n1, n2, args.tau_cl_n, batch_size=params['batch_size_1'], num_negs=None)
        loss_g = model.cl_loss_g(e1[edge_mask], e2[edge_mask], args.tau_cl_g, batch_size=params['batch_size_1'], num_negs=None)
        masked_index1 = hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, None, edge_mask1)
        masked_index2 = hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, None, edge_mask2)
        loss_m1 = model.cl_loss_m(n1, e2[edge_mask2], masked_index2, args.tau_cl_m, batch_size=params['batch_size_2'])
        loss_m2 = model.cl_loss_m(n2, e1[edge_mask1], masked_index1, args.tau_cl_m, batch_size=params['batch_size_2'])
        loss_m = (loss_m1 + loss_m2) * 0.5

        loss_cl = loss_n * args.w_cl_n + loss_g * args.w_cl_g + loss_m * args.w_cl_m
        nodes1=[]
        for x in masked_nodes1:
            nodes1+=x
        idx1=0
        edges1=[]
        for x in masked_nodes1:
            for i in range(len(x)):
                edges1.append(idx1)
            idx1+=1

        nodes2=[]
        for x in masked_nodes2:
            nodes2+=x
        idx2=0
        edges2=[]
        for x in masked_nodes2:
            for i in range(len(x)):
                edges2.append(idx2)
            idx2+=1

        queries1=scatter(src=n1[nodes1],index=torch.tensor(edges1,device=n1.device),dim=0,reduce='sum')
        prob1, gen_node1,z1 = model.generator_partial(n1,e1,queries1,masked_nodes1,[2]*len(masked_nodes1),top_p=args.top_p)

        loss_p1=model.ptr_loss_ce(prob1,masked_nodes1,gen_node1)

        queries2=scatter(src=n2[nodes2],index=torch.tensor(edges2,device=n2.device),dim=0,reduce='sum')
        prob2, gen_node2,z2 = model.generator_partial(n2,e2,queries2,masked_nodes2,[2]*len(masked_nodes2),top_p=args.top_p)

        loss_p2=model.ptr_loss_ce(prob2,masked_nodes2,gen_node2)

        loss_p=(loss_p1+loss_p2)*0.5


        loss_drop1=model.cl_loss_g(z1, e2[drop_idx1], args.tau_ptr_drop, batch_size=params['batch_size_1'], num_negs=None)
        loss_drop2=model.cl_loss_g(z2, e1[drop_idx2], args.tau_ptr_drop, batch_size=params['batch_size_1'], num_negs=None)
        loss_c=(loss_drop1+loss_drop2)*0.5

        loss_gen=args.w_ptr_ce*loss_p+args.w_ptr_drop*loss_c       
        

        
        loss=loss_cl+loss_gen
    
    loss.backward()
    optimizer.step()

    return loss.item()

def edge_prediction_eval(args,edge_split,num_splits,epochparam):
    hyperedge_index = data.edge_split[3].to(args.device)
    with torch.no_grad() : 
        model.eval()
        model.load_state_dict(epochparam)
        n,_ = model(data.features, hyperedge_index, num_nodes=data.num_nodes)
    
    valid_results, test_results, epoch=HE_predictor(n, edge_split, data.name)
    valid_auroc=valid_results[0]
    valid_ap=valid_results[1].item()
    valid_acc=valid_results[2].item()
    test_auroc=test_results[0]
    test_ap=test_results[1].item()
    test_acc=test_results[2].item()
    best_epoch=epoch

    return valid_auroc,valid_ap,valid_acc,test_auroc,test_ap,test_acc, best_epoch

def node_prediction_linear_eval(args,data_split,num_splits,epochparam):

    hyperedge_index = data.hyperedge_index
    with torch.no_grad() : 
        model.eval()
        model.load_state_dict(epochparam)
        n,_ = model(data.features, hyperedge_index, num_nodes=data.num_nodes)
    
    valid_results, test_results, epochs=NC_predictor_linear(n, data_split[:num_splits], data)
        

    return valid_results, test_results, epochs

if __name__ == '__main__':
    parser = argparse.ArgumentParser('TriCL unsupervised learning.')
    parser.add_argument('--data', type=str, default='cora_coauth', 
        choices=['citeseer_cite','cora_cite','pubmed_cite','cora_coauth','dblp_copub','dblp_coauth','aminer','imdb','modelnet_40','news','house'])
    parser.add_argument('--task', type=str, default='node', 
        choices=['edge','node'])
    parser.add_argument('--drop_hyperedge_rate', type=float, default=0.3)
    parser.add_argument('--drop_feature_rate', type=float, default=0.3)
    parser.add_argument('--drop_incidence_rate', type=float, default=0.3)
    parser.add_argument('--top_p', type=float, default=0.3)
    parser.add_argument('--epoch',type=int,default=3001)
    parser.add_argument('--ncell',type=int,default=5)
    parser.add_argument('--minhop',type=int,default=1)
    parser.add_argument('--maxhop',type=int,default=5)
    parser.add_argument('--num_layers',type=int,default=1)
    parser.add_argument('--lstm_layers',type=int,default=2)
    parser.add_argument('--hid_dim',type=int,default=256)
    parser.add_argument('--proj_dim',type=int,default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--hopmask',action='store_true')
    parser.add_argument('--cl',action='store_true')
    parser.add_argument('--cl_n',action='store_true')
    parser.add_argument('--cl_g',action='store_true')
    parser.add_argument('--cl_m',action='store_true')
    parser.add_argument('--ptr',action='store_true')
    parser.add_argument('--ptr_ce',action='store_true')
    parser.add_argument('--ptr_cl',action='store_true')
    parser.add_argument('--w_cl_n', type=float, default=0.5)
    parser.add_argument('--w_cl_m', type=float, default=0.5)
    parser.add_argument('--w_cl_g', type=float, default=0.5)
    parser.add_argument('--tau_cl_n', type=float, default=0.5)
    parser.add_argument('--tau_cl_m', type=float, default=0.5)
    parser.add_argument('--tau_cl_g', type=float, default=0.5)
    parser.add_argument('--w_ptr_drop', type=float, default=0.5)
    parser.add_argument('--w_ptr_ce', type=float, default=0.5)
    parser.add_argument('--tau_ptr_drop', type=float, default=0.5)
    parser.add_argument('--num_seeds', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--node_select',type=str,default='random')
    parser.add_argument('--mode',type=str,default='--')
    args = parser.parse_args()

    params = yaml.safe_load(open('config.yaml'))[args.data]


    data = DatasetLoader().load(args.data).to(args.device)
    
  
    fix_seed(seed=0)

    
    if args.task=="edge":

        valid_aurocs,valid_aps,valid_accs,test_aurocs,test_aps,test_accs, best_epochs=[],[],[],[],[],[],[]
        
                    
        for seed in range(args.num_seeds):
            data.edge_split=data.load_splits(seed,method=args.ns,store_use=args.storeuse)
            encoder = HyperEncoder(data.features.shape[1], args.hid_dim, args.hid_dim, args.num_layers)
            decoder = Decoder(args.hid_dim,args.ncell,args.lstm_layers)
            model = GenCL(encoder, decoder, args.proj_dim,data.features.size(1)).to(args.device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            epoch2params={}

            for epoch in tqdm(range(1, args.epoch)):
                loss= train(args)
            modelparams=copy.deepcopy(model.state_dict())
            fix_seed(0)
            valid_auroc,valid_ap,valid_acc,test_auroc,test_ap,test_acc, best_epoch = edge_prediction_eval(args,data.edge_split,args.num_seeds,modelparams)
            valid_aurocs.append(valid_auroc)
            test_aurocs.append(test_auroc)
            print(np.mean(np.array(test_auroc)))
            print(f"{np.mean(np.array(test_aurocs))*100:.1f} ± {np.std(np.array(test_aurocs))*100:.1f}\n")

        v_auroc_mean=np.mean(np.array(valid_aurocs))
        v_auroc_std=np.std(np.array(valid_aurocs))


        t_auroc_mean=np.mean(np.array(test_aurocs))
        t_auroc_std=np.std(np.array(test_aurocs))

        print(f'data:{args.data}, reuslt: {t_auroc_mean*100:.1f} ± {t_auroc_std*100:.1f}\n')
        

    if args.task=="node":
        valid_accs,test_accs,best_epochs=[],[],[]
            
        with open('data/{0}/data_split_0.01.pickle'.format(data.name), "rb") as f : 
            data_splits = pickle.load(f)
        encoder = HyperEncoder(data.features.shape[1], args.hid_dim, args.hid_dim, args.num_layers)
        decoder = Decoder(args.hid_dim,args.ncell,args.lstm_layers)
        model = GenCL(encoder, decoder, args.proj_dim,data.features.size(1)).to(args.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        epoch2params={}


        for epoch in tqdm(range(1, args.epoch)):
            
            loss= train(args)
            
        modelparams=copy.deepcopy(model.state_dict())
 
        
        fix_seed(0)
        valid_results, test_results, epoch_results = node_prediction_linear_eval(args,data_splits,args.num_seeds,modelparams)
        with torch.no_grad() : 
            model.eval()
            model.load_state_dict(modelparams)
            n,_ = model(data.features, data.hyperedge_index, num_nodes=data.num_nodes)
            
        valid_accs=np.array(valid_results)
        test_accs=np.array(test_results)

        v_acc_mean=np.mean(np.array(valid_accs))
        v_acc_std=np.std(np.array(valid_accs))


        t_acc_mean=np.mean(np.array(test_accs))
        t_acc_std=np.std(np.array(test_accs))

        
        print(f'data:{args.data}, reuslt: {t_acc_mean*100:.1f} ± {t_acc_std*100:.1f}\n')
