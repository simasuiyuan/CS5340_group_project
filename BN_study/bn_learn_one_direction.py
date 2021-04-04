import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bnlearn import *
from pgmpy.estimators import BDeuScore, K2Score, BicScore
from pgmpy.estimators import ExhaustiveSearch, HillClimbSearch
from bnlearn.bnlearn import _dag2adjmat
import pgmpy
from packaging import version
from pgmpy.estimators.HillClimbSearch import HillClimbSearch
from itertools import permutations
from collections import deque

from bnlearn.structure_learning import _SetScoringType,_is_independent

import networkx as nx
from tqdm import trange

from pgmpy.estimators import (
    StructureScore,
    StructureEstimator,
    K2Score,
    ScoreCache,
    BDeuScore,
    BicScore,
)
from pgmpy.base import DAG
from pgmpy.global_vars import SHOW_PROGRESS



def new_fit(df, methodtype='hc', scoretype='bic', black_list=None, white_list=None, bw_list_method=None, max_indegree=None, tabu_length=100, epsilon=1e-4, max_iter=1e6, root_node=None, verbose=3):
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

    out = hillclimbsearch(df,
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

    # return
    return(out)



def hillclimbsearch(df, scoretype='bic', black_list=None, white_list=None, max_indegree=None, tabu_length=100, epsilon=1e-4, max_iter=1e6, bw_list_method='enforce', verbose=3):
    out={}
    # Set scoring type
    scoring_method = _SetScoringType(df, scoretype, verbose=verbose)
    # Set search algorithm
    model = A(df, scoring_method=scoring_method)

    if bw_list_method=='enforce':
        if (black_list is not None) or (white_list is not None):
            if verbose>=3: print('[bnlearn] >Enforcing nodes based on black_list and/or white_list.')
        # best_model = model.estimate()
        best_model = model.estimate(max_indegree=max_indegree, tabu_length=tabu_length, epsilon=epsilon, max_iter=max_iter, black_list=black_list, white_list=white_list, show_progress=False)
    else:
        best_model = model.estimate(max_indegree=max_indegree, tabu_length=tabu_length, epsilon=epsilon, max_iter=max_iter, show_progress=False)

    out['model']=best_model
    out['model_edges']=best_model.edges()
    return(out)




class A(HillClimbSearch):
    #overwrite
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
            if x >y: continue #parent is x
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
                        
                        
