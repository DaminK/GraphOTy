from . import cnp
import numpy as np
import networkx as nx
import scipy as sp
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

#import ot_lib

from ot import emd, emd2, sinkhorn, gromov_wasserstein, fused_gromov_wasserstein2
from ot.unbalanced import mm_unbalanced
import seaborn as sns
import scipy.linalg as spl
from tqdm import tqdm
from karateclub import Node2Vec, Role2Vec, Graph2Vec
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from multiprocessing import Pool
from parallelbar import progress_starmap



def compute_node_dists(i,j,GMM1, GMM2,Feature1=None, Feature2=None):
    dist = cdist(GMM1['mean'], GMM2['mean'], 'sqeuclidean')
    feat_dist = None
    if Feature1 is not None:
        feat_dist = cdist(Feature1, Feature2, 'sqeuclidean')
        #hardcoded_ratio_myocard = 1.2* (10 ** -5)
        #feat_dist = feat_dist * hardcoded_ratio_myocard #/np.mean(feat_dist))*np.mean(dist)

    return i,j,dist, feat_dist

def getGMMdistance(i,j, GMM1, GMM2, regul, unbalanced, distance, covariance, eigenvalues, Feature1=None, Feature2=None, alpha=0.5, overwrite_dist=None, overwrite_feat_dist=None):
    #Wasserstein distance
    feat_dist = None
    if Feature1 is not None:
        feat_dist = cdist(Feature1, Feature2, 'sqeuclidean') if overwrite_feat_dist is None else overwrite_feat_dist

    if distance=='w':
        #TiedOT
        dist = cdist(GMM1['mean'], GMM2['mean'], 'sqeuclidean') if overwrite_dist is None else overwrite_dist
        
        #ScaledOT
        if covariance=='scaled':
            for x in range(len(GMM1['mean'])):
                for y in range(len(GMM2['mean'])):
                    d1_times_d2 = GMM1['d'][x] * GMM2['d'][y]
                    d = np.sum([(eigenvalues[l] * (GMM1['d'][x][l]+GMM2['d'][y][l]-2*np.sqrt(d1_times_d2[l])))/(d1_times_d2[l]) if d1_times_d2[l] > 0.00000001 else 0 for l in range(len(eigenvalues))])
                    dist[x,y] += d
        #FullOT            
        elif covariance=='full':
            for x in range(len(GMM1['mean'])):
                for y in range(len(GMM2['mean'])):
                    try:
                        Sigma00  = GMM1['sqrt_cov'][x]
                        Sigma010 = spl.sqrtm(Sigma00@GMM2['cov'][y]@Sigma00)
                        np.nan_to_num(Sigma010, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                        dist[x,y]+= np.trace(GMM1['cov'][x]+GMM2['cov'][y]-2*Sigma010)
                    except Exception as e:
                        print(e)
                        
        
        #Parameters for OT
        weighted_dist = alpha * dist + (1-alpha) * feat_dist if Feature1 is not None else dist
        if regul > 0:   
            if unbalanced:
                wstar = mm_unbalanced(GMM1['weight'], GMM2['weight'], weighted_dist,regul,numItermax=100*1000)
            else:
                wstar = sinkhorn(GMM1['weight'], GMM2['weight'], weighted_dist,regul,numItermax=100*1000)
        else:
            wstar     = emd(GMM1['weight'], GMM2['weight'], weighted_dist,numItermax=100*100000)         # discrete transport pl
        
        #Distances and Transport plans
        return i, j, np.sum(wstar*weighted_dist), wstar, dist, feat_dist
    elif distance=='gw' and covariance == 'tied':
        C1 = squareform(pdist(GMM1['mean'],'sqeuclidean'))
        C2 = squareform(pdist(GMM2['mean'],'sqeuclidean'))
        wstar, log =gromov_wasserstein(C1,C2,GMM1['weight'], GMM2['weight'],log=True)
        #TODO check if wstar*log dist is actually what we thought it is, 
        return i, j, np.sum(wstar*log['gw_dist']), wstar, None, None
    
    elif distance=='fgw' and covariance == 'tied':
        C1 = squareform(pdist(GMM1['mean'],'sqeuclidean'))
        C2 = squareform(pdist(GMM2['mean'],'sqeuclidean'))
        fgw_dist, log =fused_gromov_wasserstein2(feat_dist, C1,C2,GMM1['weight'], GMM2['weight'],alpha=alpha,log=True)
        #print(log["fgw_dist"])
        return i, j, fgw_dist, log["T"], None, None
        
    
    else:
        raise Exception("Not implemented.")



def emb_thread(g,G,c, d, num_inits, normalized, row_normalized, sort,embedding_name="ccb"):
    if embedding_name == 'node2vec':
        emb = Node2Vec(dimensions=c*d)
        emb.fit(G)
        embedding = np.asarray(emb.get_embedding())[:,np.newaxis,:]
    elif embedding_name == 'role2vec':
        emb = Role2Vec(dimensions=c*d)
        emb.fit(G)
        embedding = np.asarray(emb.get_embedding())[:,np.newaxis,:]
    elif embedding_name == 'ccb' or embedding_name == 'cnp':
        A = G
        A = A /np.linalg.norm(A)
        embedding = cnp(A, c=c, d=d, num_inits=num_inits, normalized=normalized, row_normalized=row_normalized, sort=sort) # shape is now nodes x instantiations x (color * depth)

    return g, embedding

def embedding_thread(g,G,embedding_name,kwargs):
    if embedding_name == 'node2vec':
        emb = Node2Vec(dimensions=kwargs["c"]*kwargs["d"])
        emb.fit(G)
        embedding = np.asarray(emb.get_embedding())[:,np.newaxis,:]
    elif embedding_name == 'role2vec':
        emb = Role2Vec(dimensions=kwargs["c"]*kwargs["d"])
        emb.fit(G)
        embedding = np.asarray(emb.get_embedding())[:,np.newaxis,:]
    elif embedding_name == 'ccb' or embedding_name == 'cnp':
        A = nx.to_numpy_array(G)
        A = A /np.linalg.norm(A)
        embedding = cnp(A, **kwargs)
    return g, embedding
    


#Attention, don't use num_workers==1, this causes the execution to stall
def getOT(Graphs, Graphs2=None, Features= None, i_instatiations=1, k_colors=3, d_depth=2, D_dimensions=None, n_graphs_per_class=1, x=None,plot=False, regul=0, unbalanced=False, normalized=True, row_normalized=True, sort=True, distance='w', covariance='tied', alpha=1., num_workers=2, embedding_method='ccb',embedding_only=False,scale_multiple_costs="mean_dist"):
    
    #TODO Currently revert back to converting to networkx cause it breaks somewhere when replacing all occurences 
    Graphs = [nx.from_numpy_array(g) for g in Graphs]
    
    graph_indcs = {0: list(range(len(Graphs))), 1: (list(range(len(Graphs), len(Graphs)+len(Graphs2))) if Graphs2 is not None else list(range(len(Graphs))))}
    if x is None:
        num_nodes = 0
        nodes_in_graph = {}
        for g,G in enumerate(Graphs + (Graphs2 if Graphs2 is not None else [])):
            nodes_in_graph[g] = list(range(num_nodes, num_nodes+len(G.nodes)))
            num_nodes += len(G.nodes)
        x = np.zeros((num_nodes, i_instatiations, k_colors*d_depth))
        
        #Features
        #if Features is not None:
        #(Graphs + (Graphs2 if Graphs2 is not None else []))

        #Structure
        if num_workers > 1:
            kwargs_list =[(g,G,embedding_method,{
                 "c":k_colors, 
                 "d":d_depth, 
                 "num_inits":i_instatiations,
                 "normalized":normalized, 
                 "row_normalized":row_normalized,
                 "sort":sort},) 
                for g,G in enumerate(Graphs + (Graphs2 if Graphs2 is not None else []))]
            map_res_cnp = progress_starmap(embedding_thread, kwargs_list, n_cpu=num_workers)
            for g, emb in map_res_cnp:
                x[nodes_in_graph[g]] = emb 
        else:
            for g,G in enumerate(Graphs + (Graphs2 if Graphs2 is not None else [])):
                A = nx.to_numpy_array(G)
                A = A /np.linalg.norm(A)
                
                params = (g,G,embedding_method,{
                 "c":k_colors, 
                 "d":d_depth, 
                 "num_inits":i_instatiations,
                 "normalized":normalized, 
                 "row_normalized":row_normalized,
                 "sort":sort},) 
                print("old call")
                emb = cnp(A, c=k_colors, d=d_depth, num_inits=i_instatiations, normalized=normalized, row_normalized=row_normalized, sort=sort) # shape is now nodes x instantiations x (color * depth)
                print("parallel call")
                g_test,emb = progress_starmap(embedding_thread, [params], n_cpu=num_workers)[0]
                #print(emb[0])
                #print(emb_test[0])
                #print("what")
                x[nodes_in_graph[g]] = emb
                    
           

    else:
        raise Exception("Not implemented, x should be None")        
        
    if D_dimensions is not None:
        x = PCA(n_components=D_dimensions).fit_transform(x.reshape(-1, k_colors*d_depth)).reshape(x.shape[:-1]+(D_dimensions,))
        
    
    if scale_multiple_costs == "mean":
        #TODO features is currently a list with graph x nodes x feats, which has homogenous shapes and leads to this
        Features= [np.asarray([node_feats for node_feats in graph_feats]) / np.average(np.asarray([node_feats for graph_feats in Features for node_feats in graph_feats])) if Features is not None else None for graph_feats in Features]
        x = x / np.mean(x)
        
    elif scale_multiple_costs == "norm":
        feat_2d_norm = np.linalg.norm(np.asarray([node_feats for graph_feats in Features for node_feats in graph_feats]),ord=2) 
        Features = [np.asarray([node_feats for node_feats in graph_feats])/ feat_2d_norm if Features is not None else None for graph_feats in Features]
        x = x / np.linalg.norm(x.reshape(len(x),-1),ord=2)
   

    GMMs = []
    for graph in graph_indcs[0] + (graph_indcs[1] if Graphs2 is not None else []):
        GMM = {
            "mean":[],
            "cov":[],
            "weight":[]
        }
        for node in nodes_in_graph[graph]:
            GMM["mean"].append(np.mean(x[node], axis=0))
            GMM["cov"].append(np.cov(x[node], rowvar=0))
            GMM["weight"].append(1)
        GMM['weight'] = np.array(GMM['weight'])/len(GMM['weight'])
    
        GMMs.append(GMM)
        GMMs_objects = []
    
    if embedding_only:
        return [GMM['mean'] for GMM in GMMs]

    eigenvalues = None
    if covariance=='scaled':
        mean_sigma = np.mean([cov for cov in GMM['cov'] for GMM in GMMs], axis=0)
    
        eigenvalues, _ = np.linalg.eig(mean_sigma)

        for GMM in GMMs:
            GMM['d'] = []
            for cov in GMM['cov']:
                GMM['d'].append(np.array([np.linalg.lstsq(cov[i].reshape(-1,1), mean_sigma[i].reshape(-1,1))[0][0][0] for i in range(len(mean_sigma))]))
                
                
    elif covariance=='full':
            for GMM in GMMs:
                GMM['sqrt_cov'] = []
                for i in range(len(GMM['cov'])):
                    GMM['sqrt_cov'].append(spl.sqrtm(GMM['cov'][i]))

    struct_cost = np.zeros((len(graph_indcs[0]), len(graph_indcs[1])),dtype=object)
    feature_cost = np.zeros((len(graph_indcs[0]), len(graph_indcs[1])),dtype=object)


    #precompute pairwise dists between nodes of all pairwise graphs in struct and feat space
    dist_res = progress_starmap(compute_node_dists, [(i,j,GMMs[graph_indcs[0][i]], GMMs[graph_indcs[1][j]], Features[graph_indcs[0][i]] if Features is not None else None, Features[graph_indcs[1][j]]  if Features is not None else None) for i in range(len(graph_indcs[0])) for j in range(len(graph_indcs[1])) if i <= j], n_cpu=num_workers)
    for i,j,dist,feat_dist in dist_res:
        struct_cost[i,j] = dist
        struct_cost[j,i] = np.transpose(dist)
        feature_cost[i,j] = feat_dist 
        feature_cost[j,i] = np.transpose(feat_dist) if feat_dist is not None else None

    if scale_multiple_costs == "dist_mean":
        #Sanity check
        hardcoded_ratio_myocard = 1.2* (10 ** -5)
        
        total_struct_cost = sum([np.sum(struct_cost[i,j]) for i in range(len(graph_indcs[0])) for j in range(len(graph_indcs[1])) ])
        total_feature_cost = sum([np.sum(feature_cost[i,j]) for i in range(len(graph_indcs[0])) for j in range(len(graph_indcs[1])) ])
        ratio = total_struct_cost / total_feature_cost



        #[np.asarray([node_feats for node_feats in graph_feats]) / np.average(np.asarray([node_feats for graph_feats in Features for node_feats in graph_feats])) if Features is not None else None for graph_feats in Features]
        feature_cost = [[feat_cost_j  *  ratio for feat_cost_j in feat_cost_i]  for feat_cost_i in feature_cost]#/np.mean(feat_dist))*np.mean(dist)

    pairwise_dist = np.zeros((len(graph_indcs[0]), len(graph_indcs[1])))
    transport_plans = np.zeros((len(graph_indcs[0]), len(graph_indcs[1])),dtype=object)

    map_res = progress_starmap(getGMMdistance, [(i,j,GMMs[graph_indcs[0][i]], GMMs[graph_indcs[1][j]], regul, unbalanced, distance, covariance, eigenvalues, Features[graph_indcs[0][i]] if Features is not None else None, Features[graph_indcs[1][j]]  if Features is not None else None, alpha, struct_cost[graph_indcs[0][i]][graph_indcs[1][j]], feature_cost[graph_indcs[0][i]][graph_indcs[1][j]] if feature_cost is not None else None) for i in range(len(graph_indcs[0])) for j in range(len(graph_indcs[1])) if i <= j], n_cpu=num_workers)


    for i,j,dist,plan,s_cost,f_cost in map_res:
        pairwise_dist[i,j] = dist
        pairwise_dist[j,i] = dist
        transport_plans[i,j] = plan
        transport_plans[j,i] = np.transpose(plan)

        #struct_cost[i,j] = np.float32(s_cost)
        #struct_cost[j,i] = np.transpose(np.float32(s_cost))
        #feature_cost[i,j] = np.float32(f_cost)
        #feature_cost[j,i] = np.transpose(np.float32(f_cost))

    if plot:
        sns.heatmap(pairwise_dist)
        plt.show()
    
    return pairwise_dist, transport_plans, struct_cost, feature_cost

