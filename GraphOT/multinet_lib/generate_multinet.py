import numpy as np
import networkx as nx

from .class_multinet import Multiplex
from .generate_GMMs import *


class SBM_Multiplex(Multiplex):

    def __init__(self,blocks,layers=None,attributes=None,feature_GMMs=None,self_connectivity = 0.5,outer_connectivity = 0.05,block_mixing_chance=0.0,plot=False,example=True,feature_noise_scale=0,adj_noise_scale=0): #TODO args, kwargs
        self.blocks = blocks
        #self.n_blocks = len(blocks)
        G = []

        if attributes is not None:
            attributes = []

        if blocks.ndim == 2:
            for l,layer_blocks in enumerate(blocks):
                #Generate Block Connectivity 
                g = nx.stochastic_block_model(layer_blocks,np.diag([self_connectivity-outer_connectivity]*len(layer_blocks))+outer_connectivity)
                G.append(nx.to_numpy_array(g))    

                #Generate Block Attributes
                if attributes is not None:
                    if feature_GMMs is None:
                        gmm, pts, true_components = random_GMM(len(layer_blocks),title=None,ax=None)
                    else:
                        gmm = feature_GMMs[l]
                    a = sample_GMM_components(gmm,layer_blocks,noise_scale=feature_noise_scale)
                    attributes.append(a)
                    self.feature_GMMs = gmm

        elif blocks.ndim == 1:
            for l in range(layers):
                #Mix blocks on layer depending on block_mixing_chance
                mixed_blocks=np.copy(self.blocks)
                for i,block in enumerate(range(self.n_blocks-1)):
                    if np.random.rand()<block_mixing_chance:
                        mixed_blocks[i+1] += mixed_blocks[i] 
                        mixed_blocks[i] = 0  

                g = nx.stochastic_block_model(mixed_blocks,np.diag([self_connectivity-outer_connectivity]*self.n_blocks)+outer_connectivity)
                g = nx.to_numpy_array(g)
                G.append(g + np.random.normal(0.5,adj_noise_scale,g.shape) )

        
        #print(np.asarray(attributes).shape)
        super().__init__(G,node_attributes=attributes,plot=plot,example=example)
