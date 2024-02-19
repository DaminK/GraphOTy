from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score

import numpy as np
import seaborn as sns

def from_dists(Graphs, labels, precomputed_dists=None, n_splits=20, method="TiedOT",n_neighbors=5,weights=None,test_size=0.2):
    predicted_labels, true_labels, scores = [],[],[]


    if n_splits>0:
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,random_state=0)
        train_test_inds = sss.split(np.zeros(len(Graphs)), labels)
    else:
        #Train = Test
        train_test_inds = [(np.arange(len(Graphs)),np.arange(len(Graphs)))]  #, np.arange(len(Graphs)))

    for i, (train_index, test_index) in enumerate(train_test_inds):
        if precomputed_dists is None:
            train_dists = compute_dists([Graphs[t] for t in train_index],None,method=method)
            test_to_train_dists = compute_dists([Graphs[t] for t in test_index],[Graphs[t] for t in train_index],method=method)
        else:
            train_dists = get_dist_precomputed(precomputed_dists, train_index, train_index)
            test_to_train_dists = get_dist_precomputed(precomputed_dists, test_index, train_index)

        
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors, metric="precomputed",weights=weights)
        neigh.fit(train_dists,[labels[t] for t in train_index])

        predicted_labels.append(neigh.predict(test_to_train_dists))
        true_labels.append(np.asarray([labels[t] for t in test_index]))
        scores.append(neigh.score(test_to_train_dists,true_labels[-1]))
        #ari.append(adjusted_rand_score(predicted_labels[-1],true_labels[-1]))

    ari = adjusted_rand_score(np.concatenate(true_labels),np.concatenate(predicted_labels))

    return predicted_labels, true_labels, scores, ari

def silhouette_score_wrapper(dists,labels):
    #wrapper function as fill diagonal is only available as inplace operation. It is needed to catch cases where due to numerical errors the distance of a graph to itself may be very close to zero, but not zero which is required by sklearn silhoute score method
    zero_dia_dists = np.copy(dists)
    np.fill_diagonal(zero_dia_dists,0)
    return silhouette_score(zero_dia_dists,labels,metric="precomputed")

def compute_dists(Graphs,Graphs2=None, method="TiedOT"):
    dist, plan = methods[method](Graphs,Graphs2)
    dist[dist<0] = 0
    return dist

def get_dist_precomputed(precomputed_dists, ind1, ind2):
    return precomputed_dists[ind1,:][:,ind2]

def plot_1split(predicted, true,title=None,ax=None):
    annot_labels_ind =np.unique(true,return_index=True)[1]
    annot_labels = true[annot_labels_ind]
    #ind
    cf_matrix = confusion_matrix(true, predicted, labels=annot_labels)
    if ax is None:
        plt.figure()
    ax = sns.heatmap(cf_matrix, annot=True, #fmt='.0', 
            cmap='Blues', xticklabels=annot_labels,yticklabels=annot_labels,ax=ax,fmt='g')
    ax.set(xlabel="Predicted Label", ylabel="True Label")
    ax.set_title(title)

def plot_table(df,tranpose=False):
    format_df = df
    format_df.set_index('method',inplace=True)
    if tranpose:
        format_df = format_df.transpose()
    display(format_df)
    print(format_df.to_latex(index=True,
                  #formatters={"name": str.upper},
                  float_format="{:.2f}".format,
    ))
    