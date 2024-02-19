import networkx as nx
import numpy as np

def permute_graph(G):
    return nx.from_numpy_array(nx.to_numpy_array(G, nodelist=np.random.permutation(list(G.nodes()))))

def swap_first_and_last(G, k=1):
    nodes = list(G.nodes())
    return nx.from_numpy_array(nx.to_numpy_array(G, nodelist=nodes[-k:] + nodes[k:-k] + nodes[:k]))

def sort_by_node_degree(G):
    nodes = list(G.nodes())
    deg = nx.degree_centrality(G)
    sorted_nodes = sorted(nodes, key=lambda x: deg[x])
    return nx.from_numpy_array(nx.to_numpy_array(G, nodelist=sorted_nodes))

def sort_by_EV(G):
    nodes = list(G.nodes())
    deg = nx.eigenvector_centrality(G)
    sorted_nodes = sorted(nodes, key=lambda x: deg[x])
    return nx.from_numpy_array(nx.to_numpy_array(G, nodelist=sorted_nodes))

def noisy(G,binary=True):
    A = nx.to_numpy_array(G)
    if binary:
        R = np.random.random(A.shape)
        A[R<0.3]=0
        A[R>0.7]=1
    else:
        A = (A + (np.random.random(A.shape)))/2
    return nx.from_numpy_array(A)

def mirror_graph(G):
    return nx.from_numpy_array(nx.to_numpy_array(G, nodelist=list(reversed(list(G.nodes())))))