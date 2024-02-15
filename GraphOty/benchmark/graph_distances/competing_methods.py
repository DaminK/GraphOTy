import numpy as np
import networkx as nx
import scipy as sp
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

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
from . import fgot_mgd

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


def getEVdist(Graphs, Graphs2=None, i_instatiations=1, k_colors=3, d_depth=2, D_dimensions=None, n_graphs_per_class=1, x=None,plot=False, regul=0, unbalanced=False, normalized=True, row_normalized=True):
    EVs = []
    max_length = np.max([len(g) for g in Graphs] + ([len(g) for g in Graphs2] if Graphs2 is not None else []))
    for i in range(len(Graphs)):
        ev = nx.eigenvector_centrality_numpy(nx.from_numpy_array(Graphs[i]))
        EVs.append(sorted([ev[i] for i in ev.keys()] + [0 for _ in range(max_length - len(ev.keys()))]))
    if Graphs2 is None:
        dist = cdist(EVs, EVs, 'euclidean')
    else:
        EVs2 = []
        for i in range(len(Graphs2)):
            ev = nx.eigenvector_centrality_numpy(nx.from_numpy_array(Graphs2[i]))
            EVs2.append(sorted([ev[i] for i in ev.keys()] + [0 for _ in range(max_length - len(ev.keys()))]))
        dist = cdist(EVs, EVs2, 'euclidean')
    if plot:
        sns.heatmap(dist)
        plt.show()
    return dist, None, None, None

def getDegreedist(Graphs, Graphs2=None, i_instatiations=1, k_colors=3, d_depth=2, D_dimensions=None, n_graphs_per_class=1, x=None,plot=False, regul=0, unbalanced=False, normalized=True, row_normalized=True):
    degs = []
    max_length = np.max([len(g) for g in Graphs] + ([len(g) for g in Graphs2] if Graphs2 is not None else []))
    for i in range(len(Graphs)):
        A = Graphs[i]
        deg = np.sum(A, axis=1)
        degs.append(sorted(deg) + [0 for _ in range(max_length - len(deg))])
    degs = np.array(degs)
    if Graphs2 is None:
        dist = cdist(degs, degs, 'euclidean')
    else:
        degs2 = []
        for i in range(len(Graphs2)):
            A = Graphs2[i]
            deg2 = np.sum(A, axis=1)
            degs2.append(sorted(deg2) + [0 for _ in range(max_length - len(deg2))])
        dist = cdist(degs, degs2, 'euclidean')
    if plot:
        sns.heatmap(dist)
        plt.show()
    return dist, None, None, None

def getNodeEmbeddingdist(Graphs, Graphs2=None, embedding_name='node2vec', meta_graph= True, i_instatiations=100, k_colors=5, d_depth=2, D_dimensions=64, n_graphs_per_class=1, x=None,plot=False, regul=0, unbalanced=False, normalized=True, row_normalized=True,distance='w'):
    Graphs = [nx.from_numpy_array(g) for g in Graphs]
    nodes_in_graph = {}
    graphs_indcs = {0: list(range(len(Graphs))), 1: (list(range(len(Graphs), len(Graphs)+len(Graphs2))) if Graphs2 is not None else list(range(len(Graphs))))}

    if meta_graph:
        MetaGraph = nx.Graph()
        for i,G in enumerate(Graphs + (Graphs2 if Graphs2 is not None else [])):
            len_before = len(MetaGraph.nodes)
            MetaGraph = nx.disjoint_union(MetaGraph, G)
            nodes_in_graph[i] = list(range(len_before, len(MetaGraph.nodes)))
        
    
    if embedding_name == 'node2vec':
        emb = Node2Vec()
        emb.fit(MetaGraph)
        embedding = emb.get_embedding()
    elif embedding_name == 'role2vec':
        emb = Role2Vec()
        emb.fit(MetaGraph)
        embedding = emb.get_embedding() 
    elif embedding_name == 'cnp':
        embedding = np.mean(cnp(MetaGraph, c=k_colors, d=d_depth, num_inits=i_instatiations, normalized=normalized), axis=1)
    else :
        raise Exception("Unknown embedding name")                                                   

    pairwise_dist = np.zeros((len(graphs_indcs[0]), len(graphs_indcs[1])))
    transport_plans = np.zeros((len(graphs_indcs[0]), len(graphs_indcs[1])),dtype=object)
    for i in tqdm(range(len(graphs_indcs[0]))):
        for j in range(len(graphs_indcs[1])):
            if i > j and Graphs2 is None:
                pairwise_dist[i,j] = pairwise_dist[j,i]
                transport_plans[i,j]= np.transpose(transport_plans[j,i])
            else:
                weight1 = np.ones((len(nodes_in_graph[graphs_indcs[0][i]]),))/len(nodes_in_graph[graphs_indcs[0][i]])
                weight2 = np.ones((len(nodes_in_graph[graphs_indcs[1][j]]),))/len(nodes_in_graph[graphs_indcs[1][j]])
                if distance=='w':
                    dist = cdist(embedding[nodes_in_graph[graphs_indcs[0][i]]], embedding[nodes_in_graph[graphs_indcs[1][j]]], 'euclidean')
                
                    if regul > 0:   
                        if unbalanced:
                #wstar     = ot.unbalanced.sinkhorn_unbalanced(pi0,pi1,M,regul,regul)
                            wstar = mm_unbalanced(weight1,weight2 , dist,regul)
                        else:
                            wstar = sinkhorn(weight1,weight2, dist,regul)
                    else:
                        wstar     = emd(weight1,weight2, dist)         # discrete transport plan
                        #distGW2   = np.sum(wstar*dist) #distance
                    #x, y = linear_sum_assignment(dist)
                    pairwise_dist[i,j] = np.sum(wstar*dist)
                    transport_plans[i,j] = wstar
                elif distance=='gw':
                    C1 = squareform(pdist(embedding[nodes_in_graph[graphs_indcs[0][i]]],'sqeuclidean'))
                    C2 = squareform(pdist(embedding[nodes_in_graph[graphs_indcs[1][j]]],'sqeuclidean'))
                    wstar, log =gromov_wasserstein(C1,C2,weight1,weight2,log=True)
                    pairwise_dist[i,j] = np.sum(wstar*log['gw_dist'])
                    transport_plans[i,j] = wstar
    if plot:
        sns.heatmap(pairwise_dist)
        plt.show()            
    
    return pairwise_dist, transport_plans, None, None
    
def getG2V_dist(Graphs, Graphs2=None, dim=50):
    Graphs = [nx.from_numpy_array(g) for g in Graphs]
    for i in range(len(Graphs)):
        #TODO MSK What was the purpose of this loop?
        nx.convert_node_labels_to_integers (Graphs[i], first_label =0 , ordering ='sorted',label_attribute ="node_type")
        mapping = dict (zip (Graphs[i], range (len (Graphs[i].nodes()) ) ) )
        Graphs[i] =  nx.relabel_nodes (Graphs[i], mapping )

    #Graphs2 = nx.convert_node_labels_to_integers (Graphs2, first_label =0 , ordering ='sorted ',label_attribute =" node_type ") if Graphs2 is not None else None
    

    graphs_indcs = {0: list(range(len(Graphs))), 1: (list(range(len(Graphs), len(Graphs)+len(Graphs2))) if Graphs2 is not None else list(range(len(Graphs))))}

    graph2vec_model = Graph2Vec(
    dimensions=dim
    )
    graph2vec_model.fit(Graphs)
    g_emb = graph2vec_model.get_embedding()


    pairwise_dist = np.zeros((len(graphs_indcs[0]), len(graphs_indcs[1])))
    for i in tqdm(range(len(graphs_indcs[0]))):
        for j in range(len(graphs_indcs[1])):
            if i > j and Graphs2 is None:
                pairwise_dist[i,j] = pairwise_dist[j,i]
            else:
                pairwise_dist[i,j] = np.linalg.norm(g_emb[graphs_indcs[0][i]]-g_emb[graphs_indcs[1][j]])
    return pairwise_dist, None, None, None

def getFixedOT(Graphs, Graphs2=None, i_instatiations=1, k_colors=3, d_depth=2, D_dimensions=None, n_graphs_per_class=1, x=None,plot=False, regul=0, unbalanced=False, normalized=True, row_normalized=True, sort=True):
    graph_indcs = {0: list(range(len(Graphs))), 1: (list(range(len(Graphs), len(Graphs)+len(Graphs2))) if Graphs2 is not None else list(range(len(Graphs))))}
    if x is None:
        num_nodes = 0
        nodes_in_graph = {}
        for g,G in enumerate(Graphs + (Graphs2 if Graphs2 is not None else [])):
            nodes_in_graph[g] = list(range(num_nodes, num_nodes+len(G.nodes)))
            num_nodes += len(G.nodes)
        x = np.zeros((num_nodes, i_instatiations, k_colors*d_depth))
        #x = np.zeros(((len(Graphs)+len(Graphs2))*n_graphs_per_class if Graphs2 is not None else len(Graphs)*n_graphs_per_class), dtype=object)
        
        for g,G in enumerate(Graphs + (Graphs2 if Graphs2 is not None else [])):
            for i in range(n_graphs_per_class):
                A = G
                A = A /np.linalg.norm(A)
                #emb = np.swapaxes([cnp(A, c=k_colors, d=d_depth, num_inits=i_instatiations) for _ in range(i_instatiations)],axis1=0,axis2=1) #inits doesnt work for now
                #expected shape of emb is nodes x instantiations x (color x depth)  (last 2 dims will be flattened after processing)
                emb = cnp(A, c=k_colors, d=d_depth, num_inits=i_instatiations, normalized=normalized, row_normalized=row_normalized, sort=sort) # shape is now nodes x instantiations x (color * depth)
                x[nodes_in_graph[g]] = emb
    else:
        raise Exception("Not implemented, x should be None")        
        
    if D_dimensions is not None:
        #raise Exception("Dimension reduction not implemented")
        #pca_x = np.concatenate([x[graph_indx][inst] for inst in range(i_instatiations) for graph_indx in graph_indcs[0] + (graph_indcs[1] if Graphs2 is None else [])], axis=0)
        #pca_x = PCA(n_components=D_dimensions).fit_transform(pca_x)
        #counter = 0
        #for graph in graph_indcs[0] + (graph_indcs[1] if Graphs2 is None else []):
            #x[graph] = pca_x[counter:counter+(x[graph].shape[0] * i_instatiations)].reshape(x[graph].shape[0], i_instatiations, D_dimensions)
            #counter += x[graph].shape[0]
        x = PCA(n_components=D_dimensions).fit_transform(x.reshape(-1, k_colors*d_depth)).reshape(x.shape[:-1]+(D_dimensions,))
        
        
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


    pairwise_dist = np.zeros((len(graph_indcs[0]), len(graph_indcs[1])))
    transport_plans = np.zeros((len(graph_indcs[0]), len(graph_indcs[1])),dtype=object)

    for i in tqdm(range(len(graph_indcs[0]))):
        for j in range(len(graph_indcs[1])):
            if i > j and Graphs2 is None:
                pairwise_dist[i,j] = pairwise_dist[j,i]
                transport_plans[i,j]= np.transpose(transport_plans[j,i])
            else:
                dist = cdist(GMMs[graph_indcs[0][i]]['mean'], GMMs[graph_indcs[1][j]]['mean'], 'sqeuclidean')
                if regul > 0:   
                    if unbalanced:
            #wstar     = ot.unbalanced.sinkhorn_unbalanced(pi0,pi1,M,regul,regul)
                        wstar = mm_unbalanced(GMMs[graph_indcs[0][i]]['weight'], GMMs[graph_indcs[1][j]]['weight'], dist,regul)
                    else:
                        wstar = sinkhorn(GMMs[graph_indcs[0][i]]['weight'], GMMs[graph_indcs[1][j]]['weight'], dist,regul)
                else:
                        wstar = emd(GMMs[graph_indcs[0][i]]['weight'], GMMs[graph_indcs[1][j]]['weight'], dist)         # discrete transport plan
                    #distGW2   = np.sum(wstar*dist) #distance
                #x, y = linear_sum_assignment(dist)
                pairwise_dist[i,j] = np.sum(wstar*dist)
                transport_plans[i,j] = wstar
    #GMMs[0]['mean']
    if plot:
        sns.heatmap(pairwise_dist)
        plt.show()
    
    return pairwise_dist, transport_plans

'''
def getG2V_dist(Graph, dim):
    graph2vec_model = Graph2Vec(
    dimensions=dim
    )
    graph2vec_model.fit(Graph)
    g_emb = graph2vec_model.get_embedding()


    pairwise_dist = np.zeros((len(Graph), len(Graph)))
    for i in range(len(Graph)):
        for j in range(len(Graph)):
            if i > j:
                pairwise_dist[i,j] = pairwise_dist[j,i]
            else:
                pairwise_dist[i,j] = np.linalg.norm(g_emb[i]-g_emb[j])
    return pairwise_dist, None
'''

def graphedit_dist(Graph):
    pairwise_dist = np.zeros((len(Graph), len(Graph)))
    for i in range(len(Graph)):
        print(i)
        for j in range(len(Graph)):
            if i > j:
                pairwise_dist[i,j] = pairwise_dist[j,i]
            else:
                pairwise_dist[i,j] = nx.graph_edit_distance(Graph[i],Graph[j])
    return pairwise_dist, None

def netdr_dist(Graph, method,**kwargs):
    pairwise_dist = np.zeros((len(Graph), len(Graph)),dtype=float)
    for i in tqdm(range(len(Graph))):
        for j in range(len(Graph)):
            if i > j:
                pairwise_dist[i,j] = pairwise_dist[j,i]
            else:
                dist = method()
                pairwise_dist[i,j] = dist.dist(Graph[i],Graph[j],**kwargs)
                print(f"{(i,j)}{pairwise_dist[i,j]}")
    return pairwise_dist, None

# G = nx.gnp_random_graph(20,0.2)
# #getScaledOT([G, G], i_instatiations=10, k_colors=3, d_depth=2, D_dimensions=None, n_graphs_per_class=1, x=None,plot=True, regul=0, unbalanced=False, normalized=True, row_normalized=True)
# #print(getTiedOT([nx.gnp_random_graph(20,0.2), nx.gnp_random_graph(30,0.2)], i_instatiations=200, D_dimensions=3))
# Graphs = [nx.gnp_random_graph(50,0.1) for _ in range(10)]
# d1 = getFullOT(Graphs, i_instatiations=1000, k_colors=3, d_depth=2)[0]
# d2 = getOT(Graphs, i_instatiations=1000, k_colors=3, d_depth=2, covariance='full')[0]
# print(d1-d2)


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.linalg as lg
import copy
import scipy.linalg as slg
#from stochastic import *

####
#GOT
def getGOT(Graphs,Graphs2=None,dist='GOT'):
    Graphs = [nx.from_numpy_array(g) for g in Graphs]

    #Graphs2 not used, only for compabilty
    pairwise_dist = np.zeros((len(Graphs), len(Graphs)))
    for i in tqdm(range(len(Graphs))):
        for j in range(len(Graphs)):
            if i > j:
                pairwise_dist[i,j] = pairwise_dist[j,i]
            else:
                if dist=="fGOT":
                    pairwise_dist[i,j] = find_trace_sink_wass_filters_reg(Graphs[i],Graphs[j], None, 'got') #find_trace_sink_wass_filters(Graphs[i],Graphs[j])
                elif dist=='stochastic-fGOT':
                     pairwise_dist[i,j] = fgot_stochastic_mgd.fgot_stochastic(get_filters(L2, 'got'), get_filters(L1, 'got'), tau=1, n_samples=5, epochs=1000, lr=50*len(L1)*len(L2), std_init = 5, loss_type = 'w_simple', tol = 1e-12, adapt_lr = True)
                elif dist=="GOT":
                    pairwise_dist[i,j] = GOT_W_dist(Graphs[i],Graphs[j])
    return pairwise_dist, None, None, None


def GOT_W_dist(A, B):
    n = len(A)

    A = nx.laplacian_matrix(A,range(n))
    A = A.todense()
    B = nx.laplacian_matrix(B,range(n))
    B = B.todense()

    l1_tilde = A + np.ones([n,n])/n #adding 1 to zero eigenvalue; does not change results, but is faster and more stable
    l2_tilde = B + np.ones([n,n])/n #adding 1 to zero eigenvalue; does not change results, but is faster and more stable
    s1_tilde = lg.inv(l1_tilde)
    s2_tilde = lg.inv(l2_tilde)
    Root_1= slg.sqrtm(s1_tilde)
    Root_2= slg.sqrtm(s2_tilde)
    return np.trace(s1_tilde) + np.trace(s2_tilde) - 2*np.trace(slg.sqrtm(Root_1 @ s2_tilde @ Root_1))

#fGOT
def get_filters(L1, method, tau = 0.2):
    if method == 'got':
        g1 = np.real(slg.sqrtm(fgot_mgd.regularise_invert_one(L1,alpha = 0.1, ones = False )))
    elif method == 'weight':
        g1 = np.diag(np.diag(L1)) - L1
    elif method == 'heat':
        g1 = slg.expm(-tau*L1)
    elif method == 'sqrtL':
        g1 = np.real(slg.sqrtm(L1))
    elif method == 'L':
        g1 = L1
    elif method == 'sq':
        g1 = L1 @ L1
    return g1
    
def find_trace_sink_wass_filters(L1, L2, epsilon = 7e-4, method = 'got', tau = 0.2):
    n = len(L1)
    m = len(L2)

    A = nx.laplacian_matrix(L1,range(n)) + np.ones([n,n])/n
    L1 = A.todense() 
    B = nx.laplacian_matrix(L2,range(m)) + np.ones([m,m])/m
    L2 = B.todense()
    

    
    p = np.repeat(1/n, n)
    q = np.repeat(1/m, m)
    norm1 = n/10
    norm2 = m/10
    max_iter = 500
    g1= get_filters(L1, method, tau)
    g2= get_filters(L2, method, tau)
    transportplan, log = fgot_mgd.fgot(g1*norm1, g2*norm2, p, q, epsilon, max_iter=max_iter, tol=1e-9, verbose=False, log=True, lapl = True)

    return log["loss"]

def find_trace_sink_wass_filters_reg(L1, L2, epsilon = 7e-4, method = 'got', tau = 0.2):
    n = len(L1)
    m = len(L2)

    A = nx.laplacian_matrix(L1,range(n))
    L1 = A.todense()
    B = nx.laplacian_matrix(L2,range(m))
    L2 = B.todense()

    p = np.repeat(1/n, n)
    q = np.repeat(1/m, m)
    max_iter = 500
    g1= get_filters(L1, method, tau)
    g2= get_filters(L2, method, tau)

    gw, log = fgot_mgd.fgot(g1, g2, p, q, epsilon*np.max(g1)*np.max(g2)/n, max_iter=max_iter, tol=1e-9, verbose=False, log=True, lapl = True)

    return log["loss"]
