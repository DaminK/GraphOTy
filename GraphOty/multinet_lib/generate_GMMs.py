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


def random_gaussian_mixture_points(n_samples=1000,true_samples_ratio=0.07, dimensions=2,n_components=20): #TODO ,max_components=20 
    #Start with random points
    rand_pts = np.random.rand(int(n_samples*true_samples_ratio),dimensions)*2-1
    #Fit gaussian mixture model
    gmm = GaussianMixture(n_components=n_components, random_state=0).fit(rand_pts)
    #Generate points from that distribution
    pts, true_components = gmm.sample(n_samples=n_samples)
    return pts, true_components

def random_GMM(n_components,n_samples=500, title=None, ax=None):
    #Generate random points from a GMM
    pts, true_components  = random_gaussian_mixture_points(n_components=n_components,n_samples=n_samples)

    #Potential noise on points


    #Fit GMM on generated points
    gmm = GaussianMixture(n_components=n_components, random_state=0).fit(pts)



    return gmm, pts, true_components

def sample_GMM_components(gmm,block_sizes,noise_scale=0):

    #convert GMM into list of Gaussian
    samples = []

    means = gmm.means_ #list of [feats]
    covs = gmm.covariances_ #list [feats,feats]
    for (m,c,b) in zip(means,covs,block_sizes):
        gaussian = GaussianMixture(n_components=1)
        gaussian.means_ = np.array([m])
        gaussian.covariances_ = np.array([c])
        gaussian.weights_ = [1]
        
        if b > 0:
            sample, _  = gaussian.sample(n_samples=b)
            sample = [s + np.random.normal(0,noise_scale,len(s)) for s in sample]
            samples.append(sample) #moved here to avoid empty elements
        else:
            sample =[]
        #samples.append(sample)
            
    #sample from each gaussian


    return samples

    #samples GMM until sufficient points are sampled from each component corresponding to sizes
    samples = {c:[] for c in range(gmm.n_components)}
    more_samples = True
    gmm.weights_=np.full((gmm.n_components),fill_value=1/gmm.n_components)
    while more_samples:
        X, y = gmm.sample(n_samples=10*np.sum(block_sizes))
        for i,component in enumerate(y):
            samples[component].append(X[i])

        if np.array([len(component_samples)>=block_sizes[component] for component, component_samples in samples.items()]).all():
            more_samples = False
        else:
            #print(samples)
            #print([len(component_samples) for component, component_samples in samples.items()])
            pass

    block_samples = []

    for b,block_size in enumerate(block_sizes):

        block= samples[b][:block_size]
        block_samples.append(block)

    return block_samples

#def GMM_to_Gaussians(gmm):
#    means = gmm.means_
#    covs = gmm.convariances_