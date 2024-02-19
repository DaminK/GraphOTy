<img src="https://github.com/DaminK/GraphOT/blob/main/doc/images/graphOty.png?raw=true" alt="GraphOty Logo" width="250" height="250">

# GraphOTy
Graph Optimal Transport as a Python Package. 

@Lennard, look into 'examples/FusedWasserstein_SyntheticMultiplex.ipynb', not cleaned up yet but already changed imports according to new structure = should be functional for you with the appropriate env (many dependencies, will provide an appropriate list for pypi dependencies). 

GraphOty has the following submodules:
*benchmark #evaluate performance of clustering etc.  (maybe "competitor" embeddings here)
*data #loading of external data formats, e.g. anndata or spm (maybe as own package?)
*embedding #CCB, CNP embedding, (maybe "competitor" embeddings here)
*graph #Multiplex Graph Classes, helper functions, graph generators
*old   #Old Code that might be integrated in the future
*OT #FusedWasserstein, FusedGromovWasserstein, 
*plot #Visualization
*tests #Unit tests, code coverage