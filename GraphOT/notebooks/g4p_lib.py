#Data handling
from mudata import MuData
import numpy as np
from anndata import AnnData
import scanpy
import pandas as pd
from sklearn.neighbors import kneighbors_graph
import networkx as nx

#Embedding
from node2vec import Node2Vec
from sklearn.manifold import TSNE

#Modelling
from sklearn.mixture import GaussianMixture

#OT
import ot
from gmmot import *

#Plot
from pymnet import *
import seaborn as sns
import matplotlib.pyplot as plt

#GRAPE
import grape
from grape import Graph
from grape.embedders import Node2VecGloVeEnsmallen

def node2vec(g,dimensions=10):
    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(g, dimensions=dimensions, walk_length=10, num_walks=100, workers=8)  # Use temp_folder for big graphs
    model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
    return (model.wv.vectors)


#TODO cast networkx grpah to grape graph after verison update
def fast_node2vec(g):
    g=grape.Graph(g)
    # Embed nodes
    node_embedding  = Node2VecGloVeEnsmallen().fit_transform(g)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
    return node_embeddings
    
def t_SNE(embedded_ND, n_components=2, plot=True, G=None):   
    embedded_2D =  TSNE(n_components=n_components, learning_rate='auto',
                    init='random', perplexity=3).fit_transform(embedded_ND)
    
    if plot and G is not None:
        plot_2D(embedded_2D, G)

    return embedded_2D

def plot_2D(embedded_2D, G):
    df = pd.DataFrame(embedded_2D, columns=['x','y'])
    df['layer'] = nx.get_node_attributes(G, 'layer').values()

    ax = sns.scatterplot(df, x='x',y='y',hue="layer") 
    plt.title("t-SNE of Embedded Nodes")
    plt.show()

def GMM_wasserstein_dist(GMM0,GMM1):
    K0,d = GMM0.means_.shape #currently only for d1 = d2
    K1, _ = GMM1.means_.shape

    pi0=GMM0.weights_
    mu0=GMM0.means_
    S0=GMM0.covariances_

    pi1=GMM1.weights_
    mu1=GMM1.means_
    S1=GMM1.covariances_

    wstar,dist = GW2(np.ravel(pi0),np.ravel(pi1),mu0.reshape(K0,d),mu1.reshape(K1,d),S0.reshape(K0,d,d),S1.reshape(K1,d,d))
    return dist



def GMM_2D(euclidean_data,n_classes=5):

    n_classes = 5
    colors = sns.color_palette("Set2",n_classes)

    def make_ellipses(gmm, ax):
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

    # Try GMMs using different types of covariances.
    estimators = {
        cov_type: GaussianMixture(
            n_components=n_classes, covariance_type=cov_type, max_iter=20, random_state=0
        )
        for cov_type in ["spherical", "diag", "tied", "full"]
    }

    n_estimators = len(estimators)

    plt.figure(figsize=(3 * n_estimators // 2, 6))
    plt.subplots_adjust(
        bottom=0.01, top=0.95, hspace=0.15, wspace=0.05, left=0.01, right=0.99
    )

    tmp_estimator = None
    for index, (name, estimator) in enumerate(estimators.items()):
        # Since we have class labels for the training data, we can
        # initialize the GMM parameters in a supervised manner.
        
        #estimator.means_init = np.array(
        #    [X_train[y_train == i].mean(axis=0) for i in range(n_classes)]
        #)

        # Train the other parameters using the EM algorithm.
        estimator.fit(euclidean_data)

        h = plt.subplot(2, n_estimators // 2, index + 1)
        make_ellipses(estimator, h)

        plt.scatter(euclidean_data[:, 0],euclidean_data[:, 1]) #,label=node_label)
        '''
        for n, color in enumerate(colors):
            data = iris.data[iris.target == n]
            plt.scatter(
                data[:, 0], data[:, 1], s=0.8, color=color, label=iris.target_names[n]
            )

        # Plot the test data with crosses
        for n, color in enumerate(colors):
            data = X_test[y_test == n]
            plt.scatter(data[:, 0], data[:, 1], marker="x", color=color)
        '''

        #y_train_pred = estimator.predict(X_train)
        #train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
        #plt.text(0.05, 0.9, "Train accuracy: %.1f" % train_accuracy, transform=h.transAxes)

        #y_test_pred = estimator.predict(X_test)
        #test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
        #plt.text(0.05, 0.8, "Test accuracy: %.1f" % test_accuracy, transform=h.transAxes)

        plt.xticks(())
        plt.yticks(())
        plt.title(name)
        tmp_estimator = estimator

    #plt.legend(scatterpoints=1, loc="lower right", prop=dict(size=12))




    plt.savefig("GMM.png")
    plt.show()

    return tmp_estimator