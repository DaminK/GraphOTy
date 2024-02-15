import numpy as np
import networkx as nx
import pandas as pd
import henhoe2vec as hh2v
from sklearn.manifold import TSNE
import ast

from .plot_multinet import plot_multiplex, plot_adj, plot_attributes, plot_structure, plot_attributes_and_structure

def test():
    pass

class Multiplex(object):
    def __init__(self,adjacencies,node_attributes=None,layer_labels=None,node_labels=None,node_attributes_labels=None,plot=False):
        """_summary_
 
        Args:
            adjacencies (_type_): _description_
            node_attributes (_type_): _description_
        """       
        self.layers = len(adjacencies)
        self.layer_label = layer_labels if layer_labels is not None else np.arange(self.layers)

        self.nodes = np.arange(np.amax([len(adj) for adj in adjacencies]))
        if isinstance(node_labels ,(list,np.ndarray)):
            self.node_labels = node_label
        elif isinstance(node_labels ,dict):
            self.node_labels = node_labels.values()[np.argsort(nodel_labels.keys())]
        elif node_labels is None:
            self.node_labels = self.nodes
        else:
            raise Exception("Only Lists and Dicts are supported as node_labels")

        
        self.node_attributes = node_attributes
        
        if isinstance(adjacencies[0],(list,np.ndarray)):
            self.adj = adjacencies
        elif isinstance(adjacencies[0], nx.Graph):
            self.adj = [nx.to_numpy_array(g) for g in adjacencies]

        self.structural_embd_2D = None
        self.structural_embd_df = None 
        self.structural_embd_array = None 

        if plot:
            self.plot()

    def plot_attributes(self, ax=None):
        plot_attributes(self.node_attributes)

    def plot_network(self, layout=nx.spring_layout, ax=None):
        plot_multiplex(self.adj,self.node_labels)

    def plot_adj(self, ax=None):
        plot_adj(self.adj)

    def plot_attr_and_struct(self):
        plot_attributes_and_structure(self.node_attributes,self.structural_embd_2D)

    def plot(self, layout=nx.spring_layout, axs=None):
        plot_multiplex(self.adj,self.node_labels)
        plot_adj(self.adj)
        
        if self.node_attributes is not None and self.structural_embd_2D is not None:
            self.plot_attr_and_struct()

    def to_edgelist(self):
        edgelist_df = pd.DataFrame(columns=["source", "source_layer", "target", "target_layer", "weight"])
        for l in range(self.layers):
            g = nx.from_numpy_array(self.adj[l])
            for u,v,a in g.edges(data=True):
                edgelist_df.loc[len(edgelist_df.index)] = [u,l,v,l,a.get("weight",1)]
            for higher_l in np.arange(self.layers-l)+l:
                for n in g.nodes:
                    edgelist_df.loc[len(edgelist_df.index)] = [n,l,n,higher_l,1]
        return edgelist_df
    
    def to_csv(self,save_path):
        if save_path.suffix in ['.edg','.csv']:
            self.to_edgelist().loc[
            :, ["source", "source_layer", "target", "target_layer", "weight"]
            ].to_csv(save_path, sep="\t", index=False, header=False)
        else:
            raise Exception(f"Saving as {save_path.suffix} is not supported")
        

    def structural_embedding(self,save_path,name,ground_truth=None,**kwargs):
        

        #Run high-dimensioanl structural embedding
        self.to_csv(save_path / f"{name}.edg")
        hh2v.henhoe2vec.run(save_path / f"{name}.edg", save_path, **kwargs, output_name=f"{name}_embedded")
        self.structural_embd_df = pd.read_csv(save_path / f"{name}_embedded.csv", sep="\t", index_col=0, header=None)
        self.structural_embd_df.index = [ast.literal_eval(tpl) for tpl in self.structural_embd_df.index] #removes nodes in index being strings -> now floats
        self.structural_embd_array = self.structural_embd_df.values

        #Compute 2D representation of struct embedding
        embd_2D = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(self.structural_embd_array)

        #Annotate 2D representation with metadata for plotting
        self.structural_embd_2D = pd.DataFrame(embd_2D,columns=['x','y'])
        self.structural_embd_2D.index = self.structural_embd_df.index
        self.structural_embd_2D["node"],self.structural_embd_2D["layer"] = zip(*self.structural_embd_2D.index.tolist())
        
        #Label pts with their true blocks
        if ground_truth is not None:
            node_to_block = [block for block in range(len(ground_truth)) for _ in range(ground_truth[block])]
            self.structural_embd_2D['block'] = [node_to_block[int(float(node))] for node in self.structural_embd_2D['node']]
        else:
            self.structural_embd_2D['block'] = ["unknown" for node in self.structural_embd_2D['node']]

        #Plot 2D embedding of highdimensional strutural representation
        if self.node_attributes is not None:
            self.plot_attr_and_struct()
        else:
            plot_structure(S_2D=self.structural_embd_2D)

        #Add metadata to high-dimensional embedding
        self.structural_embd_df = self.structural_embd_df.join(self.structural_embd_2D[['node','layer','block']])


    def fit_GMM_over_struct_and_atr(self,n_components):
        
        assert self.structural_embd_df is not None, "No strutural embedding computed, call structural_embedding first on this Multinet Object"
        
        #Convert dataframe with annotations to array layer x node_embeddings
        S = []
        for layer in self.structural_embd["layer"].unique():
            layer_S = self.structural_embd.loc[self.structural_embd['layer'] == layer]
            layer_S = layer_S.loc[:, ~data.columns.isin(['layer', 'node','block'])].values
            S.append(layer_S)


        A = self.node_attributes

        s_dim, a_dim = len(S[0][0]),len(A[0][0])

        
        combined_S_A = []
        #Concat strutural and attribute space
        for i, (layer_S,layer_A) in enumerate(zip(S,A)):
            combined_S_A.append([])
            for j, (s, a) in enumerate(zip(layer_S,layer_A)):
                combined_S_A[i].append(s+a)


        gmm = GaussianMixture(n_components=n_components, random_state=0).fit(pts)

        
    def fit_GMM(pts,n_components,title=None, ax=None):
        
        #plot_ellipses(gmm,pts,title=title,ax=ax)
        return gmm


    '''     
    def to_mudata(self):
        anndatas = {}
        for l in self.layers:
            AnnData(self.node_attributes) #node attributes = variables
            anndata.obs_names  = self.node_labels #nodes = observations
            #anndata.var_names = self.node_attributes_labels
            anndata.obsp['connectivities'] = self.adj[l] #edges = 'connectivites'
            anndatas[self.layer_labels] = anndata

        #mdata.obsp["connectivities"]= mapping   
        return MuData(anndatas)
    '''

