# Tests
import numpy as np
#TODO change import to reflect pypi package structure
from cnp import indicator_matrix_from_colors, stochastic_embedding
from scipy import sparse
import pytest

def test_indicator_matrix_from_colors():
    colors = [0, 0, 1, 1, 2, 2, 3, 3, 0, 2]
    H = indicator_matrix_from_colors(colors)
    assert H.shape == (10, 4)
    assert np.array_equal(H, np.array([[1, 0, 0, 0], 
                                 [1, 0, 0, 0], 
                                 [0, 1, 0, 0], 
                                 [0, 1, 0, 0], 
                                 [0, 0, 1, 0], 
                                 [0, 0, 1, 0],
                                 [0, 0 ,0, 1],
                                 [0, 0, 0, 1],
                                 [1, 0, 0, 0],
                                 [0, 0, 1, 0]]))
    assert type(H) == np.ndarray
    
    colors = ['a', 'b', 'a', 'c']
    H = indicator_matrix_from_colors(colors)
    assert H.shape == (4, 3)
    assert np.array_equal(H, np.array([[1, 0, 0], 
                                 [0, 1, 0], 
                                 [1, 0, 0], 
                                 [0, 0, 1]]))
    assert type(H) == np.ndarray
    assert H.dtype == np.int64
    
    rnd_colors = np.random.randint(0, 10, (1000,))
    H = indicator_matrix_from_colors(rnd_colors)
    assert H.shape == (1000, 10)
    assert np.all(np.sum(H, axis=1) == 1)
    assert np.all(np.logical_or(H == 0, H == 1))
    


def test_stochastic_embedding():
    A = sparse.random(100, 100, density=0.1, format='csr')
    emb = stochastic_embedding(A, 5, 3, algorithm_name='ccb', num_inits=10, normalization='none')
    assert type(emb) == np.ndarray
    assert emb.dtype == np.float64 or emb.dtype == np.int64
    assert emb.shape == (100, 10, 15)
    
    
    
    emb = stochastic_embedding(A, 6, 3, algorithm_name='cnp', num_inits=11, normalization='none')
    assert type(emb) == np.ndarray
    assert emb.shape == (100, 11, 18)
    assert emb.dtype == np.float64 or emb.dtype == np.int64
    
    
    with pytest.raises(ValueError):
        stochastic_embedding(A, 5, 3, algorithm_name='node2vec', num_inits=10, normalization='none')
    with pytest.raises(ValueError):
        stochastic_embedding(A, 5, 3, algorithm_name='cnp', num_inits=10, normalization='random')
    with pytest.raises(ValueError):
        stochastic_embedding(A, 1, 3, algorithm_name='cnp', num_inits=10, normalization='none')
    with pytest.raises(ValueError):
        stochastic_embedding(A, 5, 0, algorithm_name='cnp', num_inits=10, normalization='none')
    with pytest.raises(ValueError):
        stochastic_embedding(A, 5, 3, algorithm_name='cnp', num_inits=0, normalization='none')
        
    
        