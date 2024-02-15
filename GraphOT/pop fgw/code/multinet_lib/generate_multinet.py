import numpy as np
import networkx as nx

from .class_multinet import Multiplex
from .generate_GMMs import *


class SBM_Multiplex(Multiplex):

    def __init__(self,blocks,layers=None,attributes=None,self_connectivity = 0.5,outer_connectivity = 0.05,block_mixing_chance=0.0,plot=False): #TODO args, kwargs
        self.blocks = blocks
        #self.n_blocks = len(blocks)
        G = []

        if attributes is not None:
            attributes = []

        if blocks.ndim == 2:
            for layer_blocks in blocks:
                #Generate Block Connectivity 
                g = nx.stochastic_block_model(layer_blocks,np.diag([self_connectivity-outer_connectivity]*len(layer_blocks))+outer_connectivity)
                G.append(nx.to_numpy_array(g))    

                #Generate Block Attributes
                if attributes is not None:
                    gmm, pts, true_components = random_GMM(len(layer_blocks),title=None,ax=None)
                    a = sample_GMM_components(gmm,layer_blocks)
                    attributes.append(a)

        elif blocks.ndim == 1:
            for l in range(layers):
                #Mix blocks on layer depending on block_mixing_chance
                mixed_blocks=np.copy(self.blocks)
                for i,block in enumerate(range(self.n_blocks-1)):
                    if np.random.rand()<block_mixing_chance:
                        mixed_blocks[i+1] += mixed_blocks[i] 
                        mixed_blocks[i] = 0  

                g = nx.stochastic_block_model(mixed_blocks,np.diag([self_connectivity-outer_connectivity]*self.n_blocks)+outer_connectivity)
                G.append(nx.to_numpy_array(g))

        
        #print(np.asarray(attributes).shape)
        super().__init__(G,node_attributes=attributes,plot=plot,)
