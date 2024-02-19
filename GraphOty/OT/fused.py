import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from our_external_pypi_package import cnp

import numpy as np
import networkx as nx
import scipy as sp

from scipy.spatial.distance import pdist, cdist, squareform

from sklearn.decomposition import PCA

from ot import emd, emd2, sinkhorn, gromov_wasserstein, fused_gromov_wasserstein2
from ot.unbalanced import mm_unbalanced
import seaborn as sns
import scipy.linalg as spl
from tqdm import tqdm
from karateclub import Node2Vec, Role2Vec, Graph2Vec

import matplotlib.pyplot as plt

from parallelbar import progress_starmap

from itertools import zip_longest


def swap_axes_list(L):
    if L is None:
        return None
    return [[i for i in element if i is not None] for element in  list(zip_longest(*L))]


def fused_wasserstein(embeddings,features=None,alpha=None,beta=None,metric='sqeuclidean',return_plans=False,balance_weights=True,save_ram=False,threads=2):
    """_summary_
    Computes the Fused Non-Gromov Wasserstein Distance

    Args:
        embeddings (_type_): The list of the embeddings of all Graphs. The expected shape is List of length |S| (the number of structural spaces) x n_Graphs x nodes x resp. structurals space dimensions. 
        features (_type_, optional): _description_. The list of the features of all Graphs. The expected shape is List of length |F| (the number of features spaces) x n_Graphs x nodes x  resp. feature space dimensions. Defaults to None.
        alpha (_type_, optional): _description_. Defaults to None.
        beta (_type_, optional): _description_. Defaults to None.

    Returns:
        (_type_): The Fused Non-Gromov-Wasserstein Distance
    """    

    S_n = len(embeddings)
    F_n = len(features) if features is not None else 0
    n_graphs = len(embeddings[0])

    #Swap from convient shape for function call to shape better suited for computation
    embeddings, features = swap_axes_list(embeddings), swap_axes_list(features)
   
    #Compute default values for alpha & beta and assert correct length of Alpha,Beta 
    if alpha is None:
        alpha = np.ones((S_n))/S_n
    else:
        assert S_n == len(alpha), f"Dimension missmatch: {S_n} structural spaces, but alpha has length {len(alpha)}"
    if beta is None:
        if features is not None:
            beta = np.ones((F_n))/F_n
    else:
        assert F_n == len(beta), f"Dimension missmatch: {F_n} feature spaces, but beta has length {len(beta)}"
    
    #Precompute all node distances to balance the average distance across all spaces. This enables the direct use of alpha, beta to weight the importance and not the magnitude of the respective spaces.
    node_dists = np.zeros((n_graphs,n_graphs),dtype=object)
    node_dists_sum = np.zeros((n_graphs,n_graphs,(S_n+F_n)),dtype=object)
    balancing_weights = np.zeros((S_n+F_n))

    enumerator = list(zip(embeddings,features)) if F_n > 0 else list(zip(embeddings,[[]]*len(embeddings)))

    if balance_weights:
        if threads > 1:
            #Multithreading
            print("Precomputing Node Distances between Graphs in respective Structural and Feature Spaces:")
            parallel_dists = progress_starmap(compute_node_dists,[(E1+F1,E2+F2,G1,G2,save_ram) for G1, (E1,F1) in enumerate(enumerator) for G2, (E2,F2) in enumerate(enumerator) if G1 <= G2])
            for G1,G2,node_dist,node_dist_sum in parallel_dists:
                node_dists[G1,G2],node_dists_sum[G1,G2,:] = node_dist,node_dist_sum
                node_dists[G2,G1],node_dists_sum[G2,G1,:] = np.transpose(node_dist),node_dist_sum
        else:
            #Single Thread
            for G1, (E1,F1) in enumerate(enumerator):
                for G2, (E2,F2) in enumerate(enumerator):
                    if G1<G2:
                        _, _, node_dists[G1,G2],node_dists_sum[G1,G2,:] = compute_node_dists(E1+F1,E2+F2,save_ram=save_ram)
                    else:    
                        _, _, node_dists[G1,G2],node_dists_sum[G1,G2,:] = node_dists[G2,G1],node_dists_sum[G2,G1,:] 

        #We might not be able to store all precomputed node dists and just computed their total sum and balancing factors

        balancing_weights = 1 / np.sum(np.asarray(node_dists_sum, dtype=float),axis=(0,1))
        if save_ram:
            node_dists = None  
    #Iterate over all pairwise Graphs
    graph_fngw_dists = np.zeros((n_graphs,n_graphs))
    graph_fngw_plans = np.zeros((n_graphs,n_graphs),dtype=object)

    if threads > 1:
        #Multithreading
        print("Computing FNGW between all Graph Pairs")
        parallel_graph_fngw = progress_starmap(pairwise_fngw,[(E1+F1,E2+F2,alpha,beta,G1,G2,balancing_weights,return_plans,node_dists[G1,G2] if node_dists is not None else None) for G1, (E1,F1) in enumerate(enumerator) for G2, (E2,F2) in enumerate(enumerator) if G1 < G2])
        for G1,G2,graph_fngw_dist,graph_fngw_plan in parallel_graph_fngw:
            graph_fngw_dists[G1,G2],graph_fngw_plans[G1,G2] = graph_fngw_dist,graph_fngw_plan
            graph_fngw_dists[G2,G1],graph_fngw_plans[G2,G1] = graph_fngw_dist,np.transpose(graph_fngw_plan)
    else:
        #Single Thread
        for G1, (E1,F1) in enumerate(enumerator):
            for G2, (E2,F2) in enumerate(enumerator):
                if G1<G2:
                    _, _, graph_fngw_dists[G1,G2], graph_fngw_plans[G1,G2]= pairwise_fngw(E1+F1,E2+F2,alpha,beta,return_plans=return_plans,precomputed_dist=node_dists[G1,G2] if node_dists is not None else None,balancing_weights=balancing_weights)
                else:    
                    _, _, graph_fngw_dists[G1,G2], graph_fngw_plans[G1,G2] = graph_fngw_dists[G2,G1],graph_fngw_plans[G2,G1]


    if return_plans:
        return graph_fngw_dists,graph_fngw_plans, None, None
    else:
        return graph_fngw_dists


def compute_node_dists(graph1_spaces,graph2_spaces,i=None,j=None,save_ram=False,metric='sqeuclidean'):
    assert len(graph1_spaces) == len(graph2_spaces), f"Dimensions missmatch between Spaces of Graphs"
    n_spaces = len(graph1_spaces)
    nodes1 = len(graph1_spaces[0])
    nodes2 = len(graph2_spaces[0])

    Delta = np.zeros((n_spaces,nodes1,nodes2))
    for s in range(n_spaces): #enumerate(zip(graph1_spaces,graph2_spaces)):
        Delta[s,:,:] = cdist(graph1_spaces[s],graph2_spaces[s],metric=metric) 

    if save_ram:
        return i,j, None, np.sum(Delta,axis=(1,2))
    else:
        return i,j, Delta, np.sum(Delta,axis=(1,2))

def pairwise_fngw(graph1_spaces,graph2_spaces,alpha,beta,i=None,j=None,balancing_weights=None,return_plans=False,precomputed_dist=None,metric='sqeuclidean',unbalanced=False,regularization=None,):
    """_summary_

    Args:
        graph1_spaces (_type_): Spaces of Graph1 in the shape: (length |S| (the number of structural spaces) + length |F| (the number of features spaces)) x nodes x respective dimensions of space
        graph2_spaces (_type_): Spaces of Graph2 in the shape: (length |S| (the number of structural spaces) + length |F| (the number of features spaces)) x nodes x respective dimensions of space
        alpha (_type_): _description_
        beta (_type_): _description_
    """    
    assert len(graph1_spaces) == len(graph2_spaces), f"Dimensions missmatch between Spaces of Graphs"
    n_spaces = len(graph1_spaces)
    nodes1 = len(graph1_spaces[0])
    nodes2 = len(graph2_spaces[0])
    p = np.ones((nodes1))/nodes1
    q = np.ones((nodes2))/nodes2

    #Compute pairwise distances over all spaces and balance them
    Delta = np.zeros((n_spaces,nodes1,nodes2))
    Alpha_Beta = np.concatenate((alpha,beta))
    for s in range(n_spaces): #enumerate(zip(graph1_spaces,graph2_spaces)):
        Delta[s,:,:] = cdist(graph1_spaces[s],graph2_spaces[s],metric=metric) if precomputed_dist is None else precomputed_dist[s]
        if balancing_weights is not None:
            Delta[s,:,:] = Delta[s,:,:] * balancing_weights[s] * Alpha_Beta[s]   

    #Scale Delta along axis of spaces with respective weights
    weighted_Delta = np.sum(Delta,axis=0) #np.einsum('ij,jkm->ikm',np.concatenate((alpha,beta))[np.newaxis,:], Delta)[0]
    if regularization is not None:   
        if unbalanced:
            wstar = mm_unbalanced(p, q, weighted_Delta,regularization,numItermax=100*1000)
        else:
            wstar = sinkhorn(p, q, weighted_Delta,regularization,numItermax=100*1000)
    else:
        wstar     = emd(p, q, weighted_Delta,numItermax=100*100000) 

    if return_plans:
        return i,j,np.sum(wstar*weighted_Delta), wstar
    else:
        return i,j,np.sum(wstar*weighted_Delta), None
    



def fused_gromov_wasserstein(embeddings,features=None,alpha=None,beta=None,metric='sqeuclidean',return_plans=False,balance_weights=False,save_ram=False,threads=2):
    """_summary_
    Computes the Fused Gromov Wasserstein Distance

    Args:
        embeddings (_type_): The list of the embeddings of all Graphs. The expected shape is List of length |S| (the number of structural spaces) x n_Graphs x nodes x resp. structurals space dimensions. 
        features (_type_, optional): _description_. The list of the features of all Graphs. The expected shape is List of length |F| (the number of features spaces) x n_Graphs x nodes x  resp. feature space dimensions. Defaults to None.
        alpha (_type_, optional): _description_. Defaults to None.
        beta (_type_, optional): _description_. Defaults to None.

    Returns:
        (_type_): The FusedGromov-Wasserstein Distance
    """    

    S_n = len(embeddings)
    F_n = len(features) if features is not None else 0
    n_graphs = len(embeddings[0])

    #Swap from convient shape for function call to shape better suited for computation
    embeddings, features = swap_axes_list(embeddings), swap_axes_list(features)
   
    #Compute default values for alpha & beta and assert correct length of Alpha,Beta 
    if alpha is None:
        alpha = np.ones((S_n))/S_n
    else:
        assert S_n == len(alpha), f"Dimension missmatch: {S_n} structural spaces, but alpha has length {len(alpha)}"
    if beta is None:
        if features is not None:
            beta = np.ones((F_n))/F_n
    else:
        assert F_n == len(beta), f"Dimension missmatch: {F_n} feature spaces, but beta has length {len(beta)}"
    
    #Precompute all node distances to balance the average distance across all spaces. This enables the direct use of alpha, beta to weight the importance and not the magnitude of the respective spaces.
    node_dists = np.zeros((n_graphs,n_graphs),dtype=object)
    node_dists_sum = np.zeros((n_graphs,n_graphs,(S_n+F_n)),dtype=object)
    

    enumerator = list(zip(embeddings,features)) if F_n > 0 else list(zip(embeddings,[[]]*len(embeddings)))

    #Not implemented for Gromov Wasserstein yet
    if balance_weights:
        balancing_weights = np.zeros((S_n+F_n))
        pass
        '''
        if threads > 1:
            #Multithreading
            print("Precomputing Node Distances between Graphs in respective Structural and Feature Spaces:")
            parallel_dists = progress_starmap(compute_node_dists,[(E1+F1,E2+F2,G1,G2,save_ram) for G1, (E1,F1) in enumerate(enumerator) for G2, (E2,F2) in enumerate(enumerator) if G1 <= G2])
            for G1,G2,node_dist,node_dist_sum in parallel_dists:
                node_dists[G1,G2],node_dists_sum[G1,G2,:] = node_dist,node_dist_sum
                node_dists[G2,G1],node_dists_sum[G2,G1,:] = np.transpose(node_dist),node_dist_sum
        else:
            #Single Thread
            for G1, (E1,F1) in enumerate(enumerator):
                for G2, (E2,F2) in enumerate(enumerator):
                    if G1<G2:
                        _, _, node_dists[G1,G2],node_dists_sum[G1,G2,:] = compute_node_dists(E1+F1,E2+F2,save_ram=save_ram)
                    else:    
                        _, _, node_dists[G1,G2],node_dists_sum[G1,G2,:] = node_dists[G2,G1],node_dists_sum[G2,G1,:] 

        #We might not be able to store all precomputed node dists and just computed their total sum and balancing factors

        balancing_weights = 1 / np.sum(np.asarray(node_dists_sum, dtype=float),axis=(0,1))
        if save_ram:
            node_dists = None  

        '''
    else:
        balancing_weights = None    

    #Iterate over all pairwise Graphs
    graph_fngw_dists = np.zeros((n_graphs,n_graphs))
    graph_fngw_plans = np.zeros((n_graphs,n_graphs),dtype=object)

    if threads > 1:
        #Multithreading
        print("Computing FNGW between all Graph Pairs")
        parallel_graph_fngw = progress_starmap(pairwise_fgw,[(E1+F1,E2+F2,alpha,beta,G1,G2,balancing_weights,return_plans,node_dists[G1,G2] if node_dists is not None else None) for G1, (E1,F1) in enumerate(enumerator) for G2, (E2,F2) in enumerate(enumerator) if G1 < G2])
        for G1,G2,graph_fngw_dist,graph_fngw_plan in parallel_graph_fngw:
            graph_fngw_dists[G1,G2],graph_fngw_plans[G1,G2] = graph_fngw_dist,graph_fngw_plan
            graph_fngw_dists[G2,G1],graph_fngw_plans[G2,G1] = graph_fngw_dist,np.transpose(graph_fngw_plan)
    else:
        #Single Thread
        for G1, (E1,F1) in enumerate(enumerator):
            for G2, (E2,F2) in enumerate(enumerator):
                if G1<G2:
                    _, _, graph_fngw_dists[G1,G2], graph_fngw_plans[G1,G2]= pairwise_fgw(E1+F1,E2+F2,alpha,beta,return_plans=return_plans,precomputed_dist=node_dists[G1,G2] if node_dists is not None else None,balancing_weights=balancing_weights)
                else:    
                    _, _, graph_fngw_dists[G1,G2], graph_fngw_plans[G1,G2] = graph_fngw_dists[G2,G1],graph_fngw_plans[G2,G1]


    if return_plans:
        return graph_fngw_dists,graph_fngw_plans, None, None
    else:
        return graph_fngw_dists

def pairwise_fgw(graph1_spaces,graph2_spaces,alpha,beta,i=None,j=None,balancing_weights=None,return_plans=False,precomputed_dist=None,metric='sqeuclidean',unbalanced=False,regularization=None,):
    """_summary_

    Args:
        graph1_spaces (_type_): Spaces of Graph1 in the shape: (length |S| (the number of structural spaces) + length |F| (the number of features spaces)) x nodes x respective dimensions of space
        graph2_spaces (_type_): Spaces of Graph2 in the shape: (length |S| (the number of structural spaces) + length |F| (the number of features spaces)) x nodes x respective dimensions of space
        alpha (_type_): _description_
        beta (_type_): _description_
    """    
    assert len(graph1_spaces) == len(graph2_spaces), f"Dimensions missmatch between Spaces of Graphs"
    n_spaces = len(graph1_spaces)
    nodes1 = len(graph1_spaces[0])
    nodes2 = len(graph2_spaces[0])
    p = np.ones((nodes1))/nodes1
    q = np.ones((nodes2))/nodes2

    n_struct_spaces = len(alpha)
    n_feat_spaces = len(beta)

    #Compute inner distances over all strctural spaces and balance them
    C1 = np.zeros((n_struct_spaces,nodes1,nodes1))
    C2 = np.zeros((n_struct_spaces,nodes2,nodes2))

    for s in range(len(alpha)): #enumerate(zip(graph1_spaces,graph2_spaces)):
        C1[s,:,:] = squareform(pdist(graph1_spaces[s],metric=metric))  #if precomputed_dist is None else precomputed_dist[s]
        C2[s,:,:] = squareform(pdist(graph2_spaces[s],metric=metric))

        

        if balancing_weights is not None:
            C1[s,:,:] = C1[s,:,:] * balancing_weights[s] * alpha[s]   
            C2[s,:,:] = C2[s,:,:] * balancing_weights[s] * alpha[s]  
        else:
            C1[s,:,:] = C1[s,:,:] * alpha[s]   
            C2[s,:,:] = C2[s,:,:] * alpha[s]  

    #Scale along axis of spaces with respective weights
    C1 = np.sum(C1,axis=0)
    C2 = np.sum(C2,axis=0)

    M = np.zeros((n_feat_spaces,nodes1,nodes2))
    for f in range(len(beta)): #enumerate(zip(graph1_spaces,graph2_spaces)):
        M[f,:,:] = cdist(graph1_spaces[f+n_struct_spaces],graph2_spaces[f+n_struct_spaces],metric=metric) #if precomputed_dist is None else precomputed_dist[s]
        if balancing_weights is not None:
            M[f,:,:] = M[f,:,:] * balancing_weights[f+n_struct_spaces] * beta[f]  
        else:
            M[f,:,:] = M[f,:,:] * beta[f]  

    #Scale Delta along axis of spaces with respective weights
    M = np.sum(M,axis=0)


    if regularization is not None:   
        if unbalanced:
            pass
            #wstar = mm_unbalanced(p, q, weighted_Delta,regularization,numItermax=100*1000)
        else:
            pass
            #wstar = sinkhorn(p, q, weighted_Delta,regularization,numItermax=100*1000)
    else:
        fwg_dist = fused_gromov_wasserstein2(M,C1,C2,p,q,max_iter=100000)
        #wstar     = emd(p, q, weighted_Delta,numItermax=100*100000) 

    if return_plans:
        return i,j,fwg_dist , None
    else:
        return i,j,fwg_dist , None


