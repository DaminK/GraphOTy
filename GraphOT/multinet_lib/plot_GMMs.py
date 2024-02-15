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


def plot_GMM(layer_attributes,blocks):
    #only works for 2D attributes currently
    for layer_atr in layer_attributes:
        gmm = GaussianMixture(n_components=n_components, random_state=0).fit(layer_atr)
        plot_ellipses(pts=layer_atr)


def plot_ellipses(gmm, pts=None, ax=None, title=None):
        if ax is None:
            fig =plt.figure()
            ax = fig.gca()
            

        if pts is not None:
            ax.scatter(pts[:, 0],pts[:, 1],s=1,alpha=0.1) #,c='black')
        else:
            plt.show()
            
        n_classes = gmm.n_components
        colors = sns.color_palette("Set2",n_classes)
        for n, color in enumerate(colors):
            if gmm.covariance_type == "full":
                covariances = gmm.covariances_[n][:2, :2]
            elif gmm.covariance_type == "tied":
                covariances = gmm.covariances_[:2, :2]
            elif gmm.covariance_type == "diag":
                covariances = np.diag(gmm.covariances_[n][:2])
            elif gmm.covariance_type == "spherical":
                covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            ell = mpl.patches.Ellipse(
                gmm.means_[n, :2], v[0], v[1], angle=180 + angle, color=color
            )
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)
            ax.set_aspect("equal", "datalim")

        
        plt.xticks(())
        plt.yticks(())


        if title is not None:
            ax.set_title(title)