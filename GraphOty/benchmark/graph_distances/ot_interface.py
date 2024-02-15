import numpy as np

#OT
import ot
from .gmmot import *

#Plotting
import seaborn as sns
import matplotlib.pyplot as plt

def pairwise_distance(objects, distance_method = None, plot=True, log=False,**kwargs):
    if distance_method is None:
        distance_method = Wasserstein_GMM

    dist = np.zeros((len(objects),len(objects)))
    transport_plans = np.zeros((len(objects),len(objects)),dtype=object)
    for i,obj1 in enumerate(objects):
        for j,obj2 in enumerate(objects):
            
            dist[i,j], transport_plans[i,j] = distance_method(obj1,obj2,**kwargs)
            if log:
                print(f"dist {(i,j)}:{dist[i,j]}")

    if plot:
        sns.heatmap(dist) 
    return dist, transport_plans
    
        
def Wasserstein_GMM(GMM0,GMM1,regul=0,unbalanced=False,dia=False):
    K0,d = GMM0.means_.shape #currently only for d1 = d2!!
    K1, _ = GMM1.means_.shape

    pi0=GMM0.weights_
    mu0=GMM0.means_
    S0=GMM0.covariances_

    pi1=GMM1.weights_
    mu1=GMM1.means_
    S1=GMM1.covariances_

    dist, wstar = GW2(np.ravel(pi0),np.ravel(pi1),mu0.reshape(K0,d),mu1.reshape(K1,d),S0.reshape(K0,d,d),S1.reshape(K1,d,d),regul=regul,unbalanced=unbalanced,dia=dia)
    return dist, wstar

def GromovWasserstein_GMM(GMM0,GMM1,C1=None,C2=None):
    if C1 is None:
        C1 = inner_Wasserstein_GMM(GMM0)
    if C2 is None:
        C2 = inner_Wasserstein_GMM(GMM1)

    w2, log = ot.gromov.gromov_wasserstein(
        C1=C1, C2=C2, p=GMM0.weights_, q=GMM1.weights_,loss_fun =  'square_loss', verbose=False, log=True)
    
    return log['gw_dist'], w2


def inner_Wasserstein_GMM(GMM):
    pi=GMM.weights_
    mu=GMM.means_
    S =GMM.covariances_
    
    dist = np.zeros((len(mu),len(mu)))
    for i,(m0,S0) in enumerate(zip(mu,S)):
        for j,(m1,S1) in enumerate(zip(mu,S)):
            dist[i,j] = GaussianW2(m0=m0,m1=m1,Sigma0=S0,Sigma1=S1) 
    return dist

#def Wasserstein_Gaussian(G0,G1):
#    return GaussianW2(m0=m0,m1=m1,Sigma0=S0,Sigma1=S1) 