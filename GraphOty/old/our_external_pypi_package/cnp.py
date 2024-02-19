import numpy as np
import networkx as nx
import scipy.sparse as sparse
from sklearn.decomposition import PCA
import pytest

#indicator matrix is dense!
def indicator_matrix_from_colors(colors: list, 
                                 normalized=False, 
                                 padding=0):
    if not normalized:
        H = np.array([[1 if i == colors[j] else 0 for j in range(len(colors))] for i in np.unique(colors)]).transpose()
    else:
        c, counts = np.unique(colors, return_counts=True)
        H = np.array([[1/np.sqrt(counts[j]) if colors[i] == c[j] else 0 for i in range(len(colors))] for j in range(len(c))]).transpose()
        
    return H if H.shape[1] >= padding else np.concatenate([H, np.zeros((H.shape[0], padding - H.shape[1]))], axis=1)
    

# old function for backward compatibility
def cnp(A : sparse.csr_matrix, 
        c : int, 
        d : int, 
        num_inits=100, 
        H=None, 
        pca_dim=None, 
        normalized=True, 
        row_normalized=True, 
        sort=False):
    
    F = np.zeros((A.shape[0], c, d+1))
    embeddings = np.zeros((A.shape[0], num_inits, c* (d+0)))
    
    
    for it in range(num_inits):
        if H is None:
            if sort == False:
                cuts = np.random.choice(range(1, A.shape[0]), c, replace=False)
                cuts = np.sort(cuts)
                sizes = [cuts[0]] + [cuts[i] - cuts[i-1] for i in range(1, len(cuts))] + [A.shape[0] - cuts[-1]]
                colors = [i % c for i in range(len(sizes)) for size in range(sizes[i])]
                F[:,:,-1] = indicator_matrix_from_colors(colors, padding=c)
            else:
                F[:,:,-1] = indicator_matrix_from_colors(np.random.randint(0, c, (A.shape[0], )), padding=c) # random initialisation of node colors, could be improved to actually use all colors
        else:
            F[:,:,-1] = H
        # compute matrix powers
        for i in range(d):
            F[:,:,0-i-2] = A.dot(F[:,:,0-i-1])
        
        if row_normalized:    
            for i in range(d):    
                F[:,:,i] = F[:,:,i] / np.max(np.sum(F[:,:,i], axis=1))
        if normalized:
            for v in range(F.shape[0]):
                F[v,:,:] = F[v,:,:] / np.sqrt(np.sum(np.abs(F[v,:,:])**2))
        #sort for each node
        for i in range(A.shape[0]):
            np_embedding = F[i].T
            if sort:
                ind = np.lexsort(np_embedding)
            else:
                ind = np.arange(np_embedding.shape[1])
            embeddings[i, it] = np_embedding[:, ind].flatten()[:-c]
    if pca_dim is not None:
        pca = PCA(n_components=pca_dim)
        embeddings = pca.fit_transform(embeddings.reshape(-1, c*(d))).reshape(A.shape[0], num_inits, pca_dim)
    return embeddings


def stochastic_embedding(A : sparse.csr_matrix, 
                         c : int, 
                         d : int, 
                         algorithm_name='cnp',
                         num_inits=1, 
                         normalization='none', 
                         ):
    #check input
    if algorithm_name not in ['cnp', 'ccb']:
        raise ValueError('Unsupported Algorithm. algorithm_name must be one of [ccb, cnp]')
    if normalization not in ['none', 'row', 'matrix']:
        raise ValueError('Unsupported Normalization. normalization must be one of [none, row, matrix]')
    if num_inits < 1:
        raise ValueError('num_inits must be at least 1')
    if c < 2:
        raise ValueError('c must be at least 2')
    if d < 1:
        raise ValueError('d must be at least 1')
    
    
    #initialization
    F = np.zeros((A.shape[0], c, d+1))

    for it in range(num_inits):
        
        #sample random coloring matrix
        if algorithm_name == 'ccb':
            cuts = np.random.choice(range(1, A.shape[0]), c, replace=False)
            cuts = np.sort(cuts)
            sizes = [cuts[0]] + [cuts[i] - cuts[i-1] for i in range(1, len(cuts))] + [A.shape[0] - cuts[-1]]
            colors = [i % c for i in range(len(sizes)) for size in range(sizes[i])]
            F[:,:,-1] = indicator_matrix_from_colors(colors, padding=c)
        elif algorithm_name == 'cnp': 
            F[:,:,-1] = indicator_matrix_from_colors(np.random.randint(0, c, (A.shape[0], )), padding=c) # random initialisation of node colors, could be improved to actually use all colors
        else:
            raise ValueError('Unsupported Algorithm. algorithm_name must be one of [ccb, cnp]')
        # compute matrix powers
        for i in range(d):
            F[:,:,0-i-2] = A.dot(F[:,:,0-i-1])
        
        if normalization == 'row':    
            for i in range(d):    
                F[:,:,i] = F[:,:,i] / np.max(np.sum(F[:,:,i], axis=1))
                
        if normalization == 'matrix':
            for v in range(F.shape[0]):
                F[v,:,:] = F[v,:,:] / np.sqrt(np.sum(np.abs(F[v,:,:])**2))

        embeddings = np.zeros((A.shape[0], num_inits, c* (d+0)))
        
        #sort for each node
        for i in range(A.shape[0]):
            np_embedding = F[i].T
            if algorithm_name == 'cnp':        
                ind = np.lexsort(np_embedding)
            elif algorithm_name == 'ccb':
                ind = np.arange(np_embedding.shape[1])
            else:
                raise ValueError('Unsupported Algorithm. algorithm_name must be one of [ccb, cnp]')
            embeddings[i, it] = np_embedding[:, ind].flatten()[:-c]
        
    return embeddings