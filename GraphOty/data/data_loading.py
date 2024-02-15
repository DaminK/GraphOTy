import pickle
import numpy as np
from pathlib import Path
from os import path
import networkx as nx

import matplotlib.pyplot as plt

static_path =  Path.cwd().parent/"data/networks/FC"

#Helper functions
def print_dict_keys(print_dict,level=0):
    if type(print_dict) is dict:
        print(f"Level {level}:",end=" ")
        print(list(print_dict.keys()))
        print_dict_keys(next(iter(print_dict.values())),level+1)
    else:
        print(f"Example shape of Value (feature): {print_dict.shape}")

def load_FC(path, processed= True, **kwargs):
    path = static_path / path
    with open(path, 'rb') as file:
            feat_dict = pickle.load(file)
    feat_dict = feat_dict["anatomical"]
    #print_dict_keys(feat_dict)
    return feat_dict if not processed else trim_dict(feat_dict, **kwargs)


def plot_degree_dist(G,cond):
    degrees = np.sum(np.asarray([np.sum(g[0,:,:],axis=1) for g in G]),axis=0)
    plt.hist(degrees)
    plt.title(f"{len(G)} Graphs from {cond}")
    plt.show()


import copy
def trim_dict(dict,trial_mean=True,time_mean=True,squeeze=True,time_slice=None,trials_slice=None,drop=[]):
    org_dict = copy.deepcopy(dict)
    for phase, phase_dict in org_dict.items():
        if phase not in drop:
            for cond, cond_dict in phase_dict.items():
                if cond not in drop:
                    dict[phase][cond] = list(list(cond_dict.values())[0].values())[0]

                    if trials_slice is not None:
                        dict[phase][cond] = dict[phase][cond][trials_slice]

                    if time_slice is not None:
                        dict[phase][cond] = dict[phase][cond][:,time_slice,:,:]
                    
                    if trial_mean:
                        dict[phase][cond] = np.mean(dict[phase][cond],axis=0,keepdims=True)   

                    if time_mean:
                        dict[phase][cond] = np.mean(dict[phase][cond],axis=1,keepdims=True)
                    if squeeze:
                        dict[phase][cond] = np.squeeze(dict[phase][cond])
                    plot_degree_dist(dict[phase][cond],phase+cond)
                else:
                    dict[phase].pop(cond)   
        else:
            dict.pop(phase)
    print_dict_keys(dict)
    return dict

def normalize(matrix_2D,mean=False,std=False,cap_max=True,edges="positive"):
#def normalize(matrix_2D,mean=True,std=True,cap_max=False,only_positive=False):
    if mean:
        matrix_2D =  (matrix_2D - np.mean(matrix_2D)) 
    if edges=="positive":
        matrix_2D[matrix_2D<0] = 0
    elif edges=="negative":
        matrix_2D = matrix_2D * -1
        matrix_2D[matrix_2D<0] = 0
    if cap_max:
        matrix_2D = matrix_2D / np.amax(matrix_2D)
    if std:
        matrix_2D = matrix_2D / np.std(matrix_2D)
    return matrix_2D

#mausis
def load(path, norm=True, **kwargs):

#graphs_dict = ot_lib.load_FC("FC_response.npy",cond_mean=False,squeeze=False,n_trials=50)
    graphs_dict = load_FC(path, **kwargs)#,trials_slice=slice(20,30,1)) 

    labels = [f"{phase}_{cond}" for phase,phase_dict in graphs_dict.items() for cond,timepoints in phase_dict.items() for trials in timepoints for graph in trials]
    phase = [f"{phase}" for phase,phase_dict in graphs_dict.items() for cond,timepoints in phase_dict.items() for trials in timepoints for graph in trials]
    cond = [f"{cond}" for phase,phase_dict in graphs_dict.items() for cond,timepoints in phase_dict.items() for trials in timepoints for graph in trials]
    t = [float(t) for phase,phase_dict in graphs_dict.items() for cond,timepoints in phase_dict.items() for trials in timepoints for t,graph in enumerate(trials)]
    print(f"Contains nans: {np.isnan(np.asarray([graph for phase,phase_dict in graphs_dict.items() for cond,timepoints in phase_dict.items() for trials in timepoints for graph in trials])).any()}")
    Graphs = [nx.from_numpy_array(normalize(graph) if norm else graph) for phase,phase_dict in graphs_dict.items() for cond,timepoints in phase_dict.items() for trials in timepoints for graph in trials]

    if len(np.unique(phase))==1:
        phase = t
    return Graphs, labels, phase, cond
    
    

    
'''

#Translate and aggregate conditions and features into readable labels
aggr_correct_wrong = False

if aggr_correct_wrong:
    translate_cond = {
        'LR-LS': "Correct",
        'RR-RS': "Correct",
        'LR-RS': "Wrong",
        'RR-LS': "Wrong",
    }
    translate_cond = {
        'LR-LS': "Left",
        'RR-RS': "Right",
        'LR-RS': "Left",
        'RR-LS': "Right",
    }
    # translate_cond = {
    #       'Visual' : "Baseline",
    #       'Tactile': "Baseline",
    #       'Visuotactile': "Baseline",
    # }

    cond = [translate_cond[c] for c in cond]    
    
#phase = [p.split("~")[-1] for p in phase]
labels = [f"{p} {c}" for p,c in zip(phase,cond)]

print(labels)


format = True
if format:
    Trials, labels = label_formatter(phase,cond,Trials)
    #cond = labels
    phase = labels


balance = True


def label_formatter(phases,conds,graphs):
    labels = []
    Trials = []
    for phase,cond,graph in zip(phases,conds,graphs):
        if "pretrial" in phase:
            labels.append("DMN")
            Trials.append(graph)
        elif "stimulus" in phase:
            translate_cond = {
                'LR-LS': "LS",
                'RR-RS': "RS",
                'LR-RS': "Confused Stimulus",
                'RR-LS': "Confused Stimulus",
            }
            labels.append(translate_cond[cond])
            Trials.append(graph)
        elif "response" in phase:        
            translate_cond = {
                'LR-LS': "LR",
                'RR-RS': "RR",
                'LR-RS': "Wrong Response",
                'RR-LS': "Wrong Response",
            }
            labels.append(translate_cond[cond])
            Trials.append(graph)
    return Trials, labels

'''