"""
Ref: https://github.com/erdogant/bnlearn/blob/master/bnlearn/structure_learning.py
"""

from itertools import permutations

import bnlearn as bn
import matplotlib.pyplot as plt
import networkx as nx
from bnlearn.bnlearn import _dag2adjmat
from bnlearn.structure_learning import _is_independent, _SetScoringType
from pgmpy.estimators import K2Score
from pgmpy.estimators.HillClimbSearch import HillClimbSearch
from pgmpy.models import BayesianModel

import bnlearn.helpers.network as network

def modified_fit(df, methodtype='hc', scoretype='bic', black_list=None, white_list=None, bw_list_method=None, max_indegree=None, tabu_length=100, epsilon=1e-4, max_iter=1e6, root_node=None, verbose=3):

    if methodtype == 'hillclimbsearch-modified':
        if isinstance(white_list, str): white_list = [white_list]
        if isinstance(black_list, str): black_list = [black_list]
        if (white_list is not None) and len(white_list)==0: white_list = None
        if (black_list is not None) and len(black_list)==0: black_list = None
        if (methodtype!='hc') and (bw_list_method=='enforce'): raise Exception('[bnlearn] >The bw_list_method="%s" does not work with methodtype="%s"' %(bw_list_method, methodtype))

        config = {}
        config['verbose'] = verbose
        config['method'] = methodtype
        config['scoring'] = scoretype
        config['black_list'] = black_list
        config['white_list'] = white_list
        config['bw_list_method'] = bw_list_method
        config['max_indegree'] = max_indegree
        config['tabu_length'] = tabu_length
        config['epsilon'] = epsilon
        config['max_iter'] = max_iter
        config['root_node'] = root_node

        if (bw_list_method is None) and ((black_list is not None) or (white_list is not None)):
            raise Exception('[bnlearn] >Error: The use of black_list or white_list requires setting bw_list_method.')
        if df.shape[1]>10 and df.shape[1]<15:
            if config['verbose']>=2: print('[bnlearn] >Warning: Computing DAG with %d nodes can take a very long time!' %(df.shape[1]))
        if (max_indegree is not None) and methodtype!='hc':
            if config['verbose']>=2: print('[bnlearn] >Warning: max_indegree only works in case of methodtype="hc"')

        if config['verbose']>=3: print('[bnlearn] >Computing best DAG using [%s]' %(config['method']))

        # Make sure columns are of type string
        df.columns = df.columns.astype(str)
        # Filter on white_list and black_list
        out = hillclimbsearch_modified(df,
                                        scoretype=config['scoring'],
                                        black_list=config['black_list'],
                                        white_list=config['white_list'],
                                        max_indegree=config['max_indegree'],
                                        tabu_length=config['tabu_length'],
                                        bw_list_method=bw_list_method,
                                        epsilon=config['epsilon'],
                                        max_iter=config['max_iter'],
                                        verbose=config['verbose'],
                                        )
        # Setup simmilarity matrix
        adjmat = _dag2adjmat(out['model'])

        out['adjmat'] = adjmat
        out['config'] = config

        return out

    elif methodtype == 'h1-tree-k2-search':
        out = {}; best_score = -float('inf'); threshold = -7000
        for i in _h1_tree_structures(df.columns): 
            model = BayesianModel(i)
            score = K2Score(df).score(model)
            if score < threshold and score > best_score:
                best_score = score
                out['model'] = model
        print(best_score)
        out['adjmat'] = _dag2adjmat(model)
        return out

    else:
        return bn.structure_learning.fit(df, methodtype, scoretype, black_list, white_list, bw_list_method, max_indegree, tabu_length, epsilon, max_iter, root_node, verbose)



def hillclimbsearch_modified(df, scoretype='bic', black_list=None, white_list=None, max_indegree=None, tabu_length=100, epsilon=1e-4, max_iter=1e6, bw_list_method='enforce', verbose=3):
    out={}
    # Set scoring type
    scoring_method = _SetScoringType(df, scoretype, verbose=verbose)
    # Set search algorithm
    model = Modified_HCS(df, scoring_method=scoring_method)

    if bw_list_method=='enforce':
        if (black_list is not None) or (white_list is not None):
            if verbose>=3: print('[bnlearn] >Enforcing nodes based on black_list and/or white_list.')
        # best_model = model.estimate()
        best_model = model.estimate(max_indegree=max_indegree, tabu_length=tabu_length, epsilon=epsilon, max_iter=max_iter, black_list=black_list, white_list=white_list, show_progress=False)
    else:
        best_model = model.estimate(max_indegree=max_indegree, tabu_length=tabu_length, epsilon=epsilon, max_iter=max_iter, show_progress=False)

    out['model']=best_model
    out['model_edges']=best_model.edges()

    return out


class Modified_HCS(HillClimbSearch):
    
    # Overwrite
    def _legal_operations(
        self, model, score, tabu_list, max_indegree, black_list, white_list, fixed_edges
    ):

        tabu_list = set(tabu_list)

        # Step 1: Get all legal operations for adding edges.
        potential_new_edges = (
            set(permutations(self.variables, 2))
            - set(model.edges())
            - set([(Y, X) for (X, Y) in model.edges()])
        )

        for (X, Y) in potential_new_edges:
            x, y = int(X[-1]), int(Y[-1])
            if x < y: continue   # ADDED. Note: Parent is x
            # Check if adding (X, Y) will create a cycle.
            if not nx.has_path(model, Y, X):
                operation = ("+", (X, Y))
                if (
                    (operation not in tabu_list)
                    and ((X, Y) not in black_list)
                    and ((X, Y) in white_list)
                ):
                    old_parents = model.get_parents(Y)
                    new_parents = old_parents + [X]
                    if len(new_parents) <= max_indegree:
                        score_delta = score(Y, new_parents) - score(Y, old_parents)
                        yield (operation, score_delta)

        # Step 2: Get all legal operations for removing edges
        for (X, Y) in model.edges():
            operation = ("-", (X, Y))
            if (operation not in tabu_list) and ((X, Y) not in fixed_edges):
                old_parents = model.get_parents(Y)
                new_parents = old_parents[:]
                new_parents.remove(X)
                score_delta = score(Y, new_parents) - score(Y, old_parents)
                yield (operation, score_delta)

        # Step 3: Get all legal operations for flipping edges
        for (X, Y) in model.edges():
            # Check if flipping creates any cycles
            x, y = int(X[-1]), int(Y[-1])
            if x <y:continue
            if not any(
                map(lambda path: len(path) > 2, nx.all_simple_paths(model, X, Y))
            ):
                operation = ("flip", (X, Y))
                if (
                    ((operation not in tabu_list) and ("flip", (Y, X)) not in tabu_list)
                    and ((X, Y) not in fixed_edges)
                    and ((Y, X) not in black_list)
                    and ((Y, X) in white_list)
                ):
                    old_X_parents = model.get_parents(X)
                    old_Y_parents = model.get_parents(Y)
                    new_X_parents = old_X_parents + [Y]
                    new_Y_parents = old_Y_parents[:]
                    new_Y_parents.remove(X)
                    if len(new_X_parents) <= max_indegree:
                        score_delta = (
                            score(X, new_X_parents)
                            + score(Y, new_Y_parents)
                            - score(X, old_X_parents)
                            - score(Y, old_Y_parents)
                        )
                        yield (operation, score_delta)

def _h1_tree_structures(nodes): # `nodes` contains the root node as its element
    root = nodes[0]
    leaves = nodes[1::]
    num = 2**len(leaves)
    for i in range(num):
        lst = []
        for j in range(1,len(leaves)+1):
            if (i>>(j-1))&1:
                lst.append([leaves[j-1], root])
        yield lst

def bn_struct_training(sl_data, method, scoring):
    struct  = modified_fit(sl_data, methodtype = method, scoretype = scoring)
    # bn.plot(struct)   # Uncomment to save figure
    return struct




# %% PLOT
COUNT = 0
def _plot(model, pos=None, scale=1, figsize=(15, 8), verbose=3):
    """Plot the learned stucture.
    Parameters
    ----------
    model : dict
        Learned model from the .fit() function..
    pos : graph, optional
        Coordinates of the network. If there are provided, the same structure will be used to plot the network.. The default is None.
    scale : int, optional
        Scaling parameter for the network. A larger number will linearily increase the network.. The default is 1.
    figsize : tuple, optional
        Figure size. The default is (15,8).
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE
    Returns
    -------
    dict containing pos and G
        pos : list
            Positions of the nodes.
        G : Graph
            Graph model
    """
    global COUNT
    out = {}
    G = nx.DiGraph()  # Directed graph
    layout='fruchterman_reingold'

    # Extract model if in dict
    if 'dict' in str(type(model)):
        model = model.get('model', None)

    # Bayesian model
    if 'BayesianModel' in str(type(model)) or 'pgmpy' in str(type(model)):
        if verbose>=3: print('[bnlearn] >Plot based on BayesianModel')
        # positions for all nodes
        pos = network.graphlayout(model, pos=pos, scale=scale, layout=layout, verbose=verbose)
        # Add directed edge with weigth
        # edges=model.edges()
        edges=[*model.edges()]
        for i in range(len(edges)):
            G.add_edge(edges[i][0], edges[i][1], weight=1, color='k')
    elif 'networkx' in str(type(model)):
        if verbose>=3: print('[bnlearn] >Plot based on networkx model')
        G = model
        pos = network.graphlayout(G, pos=pos, scale=scale, layout=layout, verbose=verbose)
    else:
        if verbose>=3: print('[bnlearn] >Plot based on adjacency matrix')
        G = network.adjmat2graph(model)
        # Get positions
        pos = network.graphlayout(G, pos=pos, scale=scale, layout=layout, verbose=verbose)

    # Bootup figure
    plt.figure(figsize=figsize)
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.85)
    # edges
    colors = [G[u][v].get('color', 'k') for u, v in G.edges()]
    weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, arrowstyle='->', edge_color=colors, width=weights)
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
    # Get labels of weights
    # labels = nx.get_edge_attributes(G,'weight')
    # Plot weights
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
    # Making figure nice
    ax = plt.gca()
    ax.set_axis_off()

    # Uncomment and adjust the path accordingly to save BN structure graphs
    # plt.savefig("C:/Users/tclee/Desktop/result_BN/GME/hcs/graphs/{}".format(COUNT))
    COUNT += 1

    # Store
    out['pos']=pos
    out['G']=G
    return(out)

bn.__dict__['plot'] = _plot