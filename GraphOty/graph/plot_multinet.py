"""
Plot multi-graphs in 3D.
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import matplotlib.colors as cm
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

#source https://stackoverflow.com/questions/60392940/multi-layer-graph-in-networkx
class LayeredNetworkGraph(object):

    def __init__(self, graphs, node_labels=None, layout=nx.spring_layout, ax=None, scale_size=True,example=False):
        """Given an ordered list of graphs [g1, g2, ..., gn] that represent
        different layers in a multi-layer network, plot the network in
        3D with the different layers separated along the z-axis.

        Within a layer, the corresponding graph defines the connectivity.
        Between layers, nodes in subsequent layers are connected if
        they have the same node ID.

        Arguments:
        ----------
        graphs : list of networkx.Graph objects or adj matrices
            List of graphs, one for each layer.

        node_labels : dict node ID : str label or None (default None)
            Dictionary mapping nodes to labels.
            If None is provided, nodes are not labelled.

        layout_func : function handle (default networkx.spring_layout)
            Function used to compute the layout.

        ax : mpl_toolkits.mplot3d.Axes3d instance or None (default None)
            The axis to plot to. If None is given, a new figure and a new axis are created.

        """

        # book-keeping
        if isinstance(graphs[0],(list,np.ndarray)):
            self.graphs = [nx.from_numpy_array(g) for g in graphs]
        else:
            self.graphs = graphs
        self.total_layers = len(graphs)

        self.node_labels = node_labels if not example else None
        print(f"example {example}")
        self.layout = layout

        if ax:
            self.ax = ax    
        else:
            size = 10 +  self.graphs[0].number_of_nodes() / 50
            fig = plt.figure(figsize=(size,size),)
            self.ax = fig.add_subplot(111, projection='3d',computed_zorder=False)

        self.ax.set_axis_off()

        # create internal representation of nodes and edges
        self.get_nodes()
        self.get_edges_within_layers()
        self.get_edges_between_layers()

        # compute layout and plot
        self.get_node_positions()
        self.draw()


    def get_nodes(self):
        """Construct an internal representation of nodes with the format (node ID, layer)."""
        self.nodes = []
        for z, g in enumerate(self.graphs):
            self.nodes.extend([(node, z) for node in g.nodes()])


    def get_edges_within_layers(self):
        """Remap edges in the individual layers to the internal representations of the node IDs."""
        self.edges_within_layers = []
        for z, g in enumerate(self.graphs):
            self.edges_within_layers.extend([((source, z), (target, z)) for source, target in g.edges()])


    def get_edges_between_layers(self):
        """Determine edges between layers. Nodes in subsequent layers are
        thought to be connected if they have the same ID."""
        self.edges_between_layers = []
        for z1, g in enumerate(self.graphs[:-1]):
            z2 = z1 + 1
            h = self.graphs[z2]
            shared_nodes = set(g.nodes()) & set(h.nodes())
            self.edges_between_layers.extend([((node, z1), (node, z2)) for node in shared_nodes])


    def get_node_positions(self, *args, **kwargs):
        """Get the node positions in the layered layout."""
        # What we would like to do, is apply the layout function to a combined, layered network.
        # However, networkx layout functions are not implemented for the multi-dimensional case.
        # Futhermore, even if there was such a layout function, there probably would be no straightforward way to
        # specify the planarity requirement for nodes within a layer.
        # Therefor, we compute the layout for the full network in 2D, and then apply the
        # positions to the nodes in all planes.
        # For a force-directed layout, this will approximately do the right thing.
        # TODO: implement FR in 3D with layer constraints.

        [nx.to_numpy_array(g) for g in self.graphs]
        composed_adj = np.sum([nx.to_numpy_array(g) for g in self.graphs], axis=0)
        composed_adj = np.log2(composed_adj+1,where=(composed_adj!=0))
        print(composed_adj)
        composition = nx.from_numpy_array(composed_adj)

        #composition = self.graphs[0]
        #for h in self.graphs[1:]:
        #    composition = nx.compose(composition, h)

        pos = self.layout(composition, *args, **kwargs)
        

        self.node_positions = dict()
        for z, g in enumerate(self.graphs):
            pos_g = self.layout(g, pos=pos, *args, **kwargs)

            self.node_positions.update({(node, z) : (*pos_g[node], z) for node in g.nodes()})


    def draw_nodes(self, nodes, *args, **kwargs):
        x, y, z = zip(*[self.node_positions[node] for node in nodes])
        self.ax.scatter(x, y, z,  *args, **kwargs)


    def draw_edges(self, edges, *args, **kwargs):
        segments = [(self.node_positions[source], self.node_positions[target]) for source, target in edges]
        #z_orders = [z_in for (_, z_in), _ in edges]
        line_collection = Line3DCollection(segments, *args, **kwargs)
        self.ax.add_collection3d(line_collection)


    def get_extent(self, pad=0.1):
        xyz = np.array(list(self.node_positions.values()))
        xmin, ymin, _ = np.min(xyz, axis=0)
        xmax, ymax, _ = np.max(xyz, axis=0)
        dx = xmax - xmin
        dy = ymax - ymin
        return (xmin - pad * dx, xmax + pad * dx), \
            (ymin - pad * dy, ymax + pad * dy)


    def draw_plane(self, z, *args, **kwargs):
        (xmin, xmax), (ymin, ymax) = self.get_extent(pad=0.1)
        u = np.linspace(xmin, xmax, 10)
        v = np.linspace(ymin, ymax, 10)
        U, V = np.meshgrid(u ,v)
        W = z * np.ones_like(U)
        self.ax.plot_surface(U, V, W, *args, **kwargs)


    def draw_node_labels(self, node_labels, *args, **kwargs):
        for node, z in self.nodes:
            if node in node_labels:
                self.ax.text(*self.node_positions[(node, z)], node_labels[node], zorder=3.5+z,*args, **kwargs)


    def draw(self):

        self.draw_edges(self.edges_within_layers,  color='k', alpha=0.4, linestyle='-', linewidth=1,zorder=2)
        self.draw_edges(self.edges_between_layers, color='k', alpha=0.8, linestyle=':', linewidth=1,zorder=2)

        for z in range(self.total_layers):
            self.draw_plane(z, alpha=0.4, zorder=2.5+z)
            self.draw_nodes([node for node in self.nodes if node[1]==z], s=200, zorder=3+z,alpha=1,edgecolors='k')

        if self.node_labels is not None:
            self.draw_node_labels(self.node_labels,
                                  fontsize=8,
                                  horizontalalignment='center',
                                  verticalalignment='center')
            

def plot_multiplex(G, node_labels=None, example=True):
    #
    #ax = fig.add_subplot(111, projection='3d')
    LayeredNetworkGraph(G, node_labels={g:g for g in G[0]} if node_labels is None else node_labels, layout=nx.spring_layout, example=example)
    
    #plt.show()

def plot_adj(G,example=False):
    mpl_scatter_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    figsize = 10 +  len(G[0]) / 50
    fig, axs = plt.subplots(ncols=len(G),figsize=(figsize,figsize / (len(G)-1)))
    for i,adj in enumerate(G):
        res = sns.heatmap(adj,ax=axs[len(G)-i-1],square=True,cbar=False,xticklabels=False, yticklabels=False,cmap=create_linear_cmap(mpl_scatter_colors [i]))
        for _, spine in res.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(5)
    plt.show


def plot_attributes_and_structure(A,S):
    figsize = 10 +  len(A[0]) / 50
    fig, (ax_S,ax_A) = plt.subplots(ncols=2,figsize=(figsize,figsize / (len(A)-1)))
    #Test
    #(ax_S,ax_A)  = None,None
    plot_structure(S,ax_S)
    plot_attributes(A,ax_A,individual_layers=False)

    plt.show()


def plot_attributes(A,ax=None,individual_layers=True,direction="horizontal",example=False):
    #only works for 2D attributes

    #A is in form layer x block x pts

    df_list = [{"x":pt[0],"y":pt[1],"block":b,"layer":l} for l,atr in enumerate(A) for b,pts in enumerate(atr) for pt in pts]

    df =  pd.json_normalize(df_list)
    layers = df['layer'].unique()
    n_layers = len(layers)

    if ax is None:
        if (individual_layers and direction=="horizontal"):
            fig, ax = plt.subplots(ncols=len(layers),figsize=(len(layers)*5, 5))
        elif (individual_layers and direction=="vertical"):
            fig, ax = plt.subplots(nrows=len(layers),figsize=(5,len(layers)*5))
        else:
            fig, ax = plt.subplots(figsize=(len(layers)*5, 5))

    if individual_layers:
        for i in layers:
            #Warning, layers inverted as i counted them from top to bottom and its easier than chaning the plot func
            sns.scatterplot(df[df['layer']==i], x='x',y='y',hue='block',palette="deep",ax=ax[n_layers-i-1],s=50)
            #ax[n_layers-i-1].title.set_text(f"Layer {n_layers-i}")
            if example:
                ax[n_layers-i-1].tick_params(left=False,bottom=False)
                ax[n_layers-i-1].set(xticklabels=[],yticklabels=[])
                ax[n_layers-i-1].set_xlabel(ax[n_layers-i-1].get_xlabel(), fontdict={'weight': 'bold','size':15})
                ax[n_layers-i-1].set_ylabel(ax[n_layers-i-1].get_ylabel(), fontdict={'weight': 'bold','size':15})
            for _, spine in ax[n_layers-i-1].spines.items():
                spine.set_visible(True)
                spine.set_linewidth(5)
    else:
        ax = sns.scatterplot(df,ax=ax, x='x',y='y',hue='block',style='layer',palette="deep")
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
  


def plot_structure(S_2D,ax=None):
    #df_embed = pd.DataFrame(S_2D,columns=['x','y'])
    if ax is None:
        plt.figure()
    ax = sns.scatterplot(S_2D,x='x',y='y',hue='block',style='layer',palette="deep",ax=ax) #,ax=axs[i])

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    


def create_linear_cmap(color):
    map_colors = [(1.0,1.0,1.0),color]
    return cm.LinearSegmentedColormap.from_list("my_cmap", map_colors)