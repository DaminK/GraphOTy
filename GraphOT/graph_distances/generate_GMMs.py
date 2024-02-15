#Data handling
import numpy as np

import pandas as pd
from sklearn.neighbors import kneighbors_graph
import networkx as nx

#Embedding
#from node2vec import Node2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
#Modelling
from sklearn.mixture import GaussianMixture
import sklearn
from copy import deepcopy



#Plot
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


def random_gaussian_mixture_points(n_samples_per_comp=20, n_dims=2,n_components=20): #TODO ,max_components=20 
    #Start with random points
    rand_pts = np.random.rand(int(n_samples_per_comp*n_components),n_dims)

    #Fit gaussian mixture model
    gmm = GaussianMixture(n_components=n_components, random_state=0).fit(rand_pts)

    #Generate points from that distribution
    pts, true_components = gmm.sample(n_samples=n_samples_per_comp*n_components)

    return pts, true_components 

def random_GMM(n_components,n_samples_per_comp=20,n_dims=2,prior_GMM=None,noise_sigma=0.01): #,n_samples=500, title=None, ax=None):
    #Generate random points from a GMM
    if prior_GMM is None:
        pts, _  = random_gaussian_mixture_points(n_components=n_components, n_samples_per_comp=n_samples_per_comp, n_dims = n_dims) #,n_samples=n_samples)
    else:
        pts, _ = prior_GMM.sample(n_samples=n_samples_per_comp*n_components)

    #Potential noise on points
    for j,pt in enumerate(pts):
        noise = np.random.normal(0, noise_sigma, pt.shape)
        pts[j] = pt + noise
    
    #Fit GMM on generated points
    gmm = GaussianMixture(n_components=n_components, random_state=0).fit(pts)

    # Plot if 2D
    # plot_ellipses(gmm,pts,title=title,ax=ax)

    return gmm #, pts, true_components

def sample_GMM_components(gmm,block_sizes):
    samples = []
    means = gmm.means_ #list of [feats]
    covs = gmm.covariances_ #list [feats,feats]

    #converts GMM into list of Gaussians (GMM with 1 comp) and samples them
    for (m,c,b) in zip(means,covs,block_sizes):
        gaussian = GaussianMixture(n_components=1)
        gaussian.means_ = np.array([m])
        gaussian.covariances_ = np.array([c])
        gaussian.weights_ = [1]

        samples.append(gaussian.sample(n_samples=b) if b > 0 else [])

    return samples
