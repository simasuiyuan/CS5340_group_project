import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chisquare
from warnings import warn
from functools import lru_cache
from scipy.special import gammaln
from math import lgamma, log
from pgmpy.models import BayesianModel
from functools import partial 

from IPython.display import Markdown, display
import networkx as nx
import pylab as plt
def printmd(string):
    display(Markdown(string))
import warnings
warnings.filterwarnings('ignore')


class BaseEstimator(object):
    def __init__(self, data=None, state_names=None, complete_samples_only=True):
        self.data = data
        if self.data is not None:
            self.complete_samples_only = complete_samples_only
            self.variables = list(data.columns.values)
            if not isinstance(state_names, dict):
                self.state_names = {
                    var: self._collect_state_names(var) for var in self.variables
                }
            else:
                self.state_names = dict()
                for var in self.variables:
                    if var in state_names:
                        if not set(self._collect_state_names(var)) <= set(
                            state_names[var]
                        ):
                            raise ValueError(
                                f"Data contains unexpected states for variable: {var}."
                            )
                        self.state_names[var] = state_names[var]
                    else:
                        self.state_names[var] = self._collect_state_names(var)
                        
    def _collect_state_names(self, variable):
        "Return a list of states that the variable takes in the data"
        states = sorted(list(self.data.loc[:, variable].dropna().unique()))
        return states
    
    def convert_args_tuple(func):
        def _convert_param_to_tuples(
            obj, variable, parents=tuple(), complete_samples_only=None
        ):
            parents = tuple(parents)
            return func(obj, variable, parents, complete_samples_only)

        return _convert_param_to_tuples
    
    @convert_args_tuple
    @lru_cache(maxsize=2048)
    def state_counts(self, variable, parents=[], complete_samples_only=None):
        """
        Return counts how often each state of 'variable' occurred in the data.
        If a list of parents is provided, counting is done conditionally
        for each state configuration of the parents.
        Returns
        -------
        state_counts: pandas.DataFrame
            Table with state counts for 'variable'
        """
        parents = list(parents)

        # default for how to deal with missing data can be set in class constructor
        if complete_samples_only is None:
            complete_samples_only = self.complete_samples_only
        # ignores either any row containing NaN, or only those where the variable or its parents is NaN
        data = (
            self.data.dropna()
            if complete_samples_only
            else self.data.dropna(subset=[variable] + parents)
        )

        if not parents:
            # count how often each state of 'variable' occured
            state_count_data = data.loc[:, variable].value_counts()
            state_counts = (
                state_count_data.reindex(self.state_names[variable])
                .fillna(0)
                .to_frame()
            )

        else:
            parents_states = [self.state_names[parent] for parent in parents]
            # count how often each state of 'variable' occured, conditional on parents' states
            state_count_data = (
                data.groupby([variable] + parents).size().unstack(parents)
            )
            if not isinstance(state_count_data.columns, pd.MultiIndex):
                state_count_data.columns = pd.MultiIndex.from_arrays(
                    [state_count_data.columns]
                )
                
            row_index = self.state_names[variable]
            column_index = pd.MultiIndex.from_product(parents_states, names=parents)
            state_counts = state_count_data.reindex(
                index=row_index, columns=column_index
            ).fillna(0)

        return state_counts
    
    
class K2Score(BaseEstimator):
    def __init__(self, data, **kwargs):
        super(K2Score, self).__init__(data, **kwargs)
        
    def K2_score_single(self, variable, parents, is_print=False):
        var_states = self.state_names[variable]
        var_cardinality = len(var_states) # number of element in a (undefined set) axiomatic set theory
        state_counts = self.state_counts(variable, parents)
        num_parents_states = float(state_counts.shape[1])
        counts = np.asarray(state_counts)
        
        if is_print:
            printmd(f'r_i = :{var_cardinality}')
            printmd(f'q_i = :{num_parents_states}')
            printmd('state_counts($N_{ijk}$):'+str(counts))
            display(state_counts.head(5))
            print('-'*100)
            
        log_gamma_counts = np.zeros_like(counts, dtype=np.float_)
        
        # Compute log(gamma(counts + 1))
        gammaln(counts + 1, out=log_gamma_counts)
        if is_print: 
            printmd("Let $\eta_{ijk} = 1$")
            printmd("$log(\Gamma(N_{ijk} + \eta_{ijk}) / \Gamma(\eta_{ijk}))$ = $log(\Gamma(N_{ijk} + 1) / 1)$ = ")
            print(f"log_gamma_counts = \n {log_gamma_counts}")
            print('-'*100)
        
        # Compute the log-gamma conditional sample size
        log_gamma_conds = np.sum(counts, axis=0, dtype=np.float_)
        gammaln(log_gamma_conds + var_cardinality, out=log_gamma_conds)
        if is_print: 
            printmd("$N_{ij} = \sum_k(N_{ijk})$")
            printmd("$\eta_{ij} = \sum_k(\eta_{ijk}) = k * 1$")
            printmd("$log(\eta_{ij}) / \Gamma(N_{ij} + \eta_{ij}))$ = $ -log(\Gamma(N_{ij} + k))$ = ")
            print(f"log_gamma_conds(-ve sign will be subtract in score computation) = \n {log_gamma_conds}")
            print('-'*100)
        
        # Compute the log P(G)
        log_PG = num_parents_states * lgamma(var_cardinality)
        if is_print: 
            print(f"num_parents_states = {num_parents_states}")
            print(f"var_cardinality = {var_cardinality}")
            printmd("$P(G) = no\_of\_paremts * log(\Gamma(no\_of\_states)) $")
            print(f"log(P(G)) = num_parents_states * lgamma(var_cardinality) = {log_PG}") 
            print('-'*100)
        score = (
            np.sum(log_gamma_counts)
            - np.sum(log_gamma_conds)
            + num_parents_states * lgamma(var_cardinality)
        )
        if is_print: 
            print(f"np.sum(log_gamma_counts) - np.sum(log_gamma_conds) + log_PG:")
        return score
    
    def score(self, model,is_print=False):
        score = 0
        for node in model.nodes():
            if is_print:
                printmd(f"## node: {node}")
                printmd(f"### parent: {list(model.predecessors(node))}")
                printmd(f"score: {self.K2_score_single(node, model.predecessors(node),is_print=is_print)}")
                print(f"="*100)
            score += self.K2_score_single(node, model.predecessors(node))
        return score
    
class BDeuScore(BaseEstimator):
    def __init__(self, data, equivalent_sample_size=10, **kwargs):
        self.equivalent_sample_size = equivalent_sample_size
        super(BDeuScore, self).__init__(data, **kwargs)
        
    def BDeu_score_single(self, variable, parents, is_print=False):
        var_states = self.state_names[variable]
        var_cardinality = len(var_states) # number of element in a (undefined set) axiomatic set theory
        state_counts = self.state_counts(variable, parents)
        num_parents_states = float(state_counts.shape[1])
        counts = np.asarray(state_counts)
        
        if is_print:
            printmd(f'r_i = :{var_cardinality}')
            printmd(f'q_i = :{num_parents_states}')
            printmd('state_counts($N_{ijk}$):'+str(counts))
            display(state_counts.head(2))
            print('-'*100)
            
        log_gamma_counts = np.zeros_like(counts, dtype=np.float_)
        
        
        alpha = self.equivalent_sample_size / num_parents_states
        beta = self.equivalent_sample_size / counts.size
        if is_print:
            printmd("##### Let $alpha = {N'}/q_{i}$ = " + str(alpha))
            printmd("##### Let $ beta = {N'}/(r_{i}q_{i})$ = " + str(beta))
            print('-'*100)
        
        
        # Compute log(gamma(counts + beta))
        gammaln(counts + beta, out=log_gamma_counts)
        if is_print: 
            printmd("$log(\Gamma(N_{ijk} + beta))$ = ")
            print(f"log_gamma_counts = \n {log_gamma_counts}")
            print('-'*100)
        
        # Compute the log-gamma conditional sample size
        log_gamma_conds = np.sum(counts, axis=0, dtype=np.float_)
        gammaln(log_gamma_conds + alpha, out=log_gamma_conds)
        if is_print: 
            printmd("$log(\Gamma(N_{ij} + alpha))$ = $ log(\Gamma(N_{ij} + k))$")
            print(f"log_gamma_conds = \n {log_gamma_conds}")
            print('-'*100)
        
        
        score = (
            np.sum(log_gamma_counts)
            - np.sum(log_gamma_conds)
            + num_parents_states * lgamma(alpha)
            - counts.size * lgamma(beta)
        )
        if is_print: 
            print(f"np.sum(log_gamma_counts) - np.sum(log_gamma_conds) + num_parents_states * lgamma(alpha) - counts.size * lgamma(beta):")
        return score
    
    def score(self, model,is_print=False):
        score = 0
        for node in model.nodes():
            if is_print:
                printmd(f"## node: {node}")
                printmd(f"### parent: {list(model.predecessors(node))}")
                printmd(f"score: {self.BDeu_score_single(node, model.predecessors(node),is_print=is_print)}")
                print(f"="*100)
            score += self.BDeu_score_single(node, model.predecessors(node))
        return score
    
    
class BicScore(BaseEstimator):
    def __init__(self, data, **kwargs):
        super(BicScore, self).__init__(data, **kwargs)
        
    def Bic_score_single(self, variable, parents, is_print=False):
        var_states = self.state_names[variable]
        var_cardinality = len(var_states) # number of element in a (undefined set) axiomatic set theory
        state_counts = self.state_counts(variable, parents)
        num_parents_states = float(state_counts.shape[1])
        counts = np.asarray(state_counts)
        sample_size = len(self.data)
        
        if is_print:
            printmd(f'r_i = :{var_cardinality}')
            printmd(f'q_i = :{num_parents_states}')
            printmd('state_counts($N_{ijk}$):'+str(counts))
            printmd('sample_size($N$):'+str(sample_size))
            display(state_counts.head(2))
            print('-'*100)
            
        log_likelihoods = np.zeros_like(counts, dtype=np.float_)
        
        # Compute the log-counts
        np.log(counts, out=log_likelihoods, where=counts > 0)
        if is_print: 
            printmd("$log(N_{ijk})(>0)$ = ")
            print(f"{log_likelihoods}")
            print('-'*100)
        
        # Compute the log-conditional sample size
        log_conditionals = np.sum(counts, axis=0, dtype=np.float_)
        np.log(log_conditionals, out=log_conditionals, where=log_conditionals > 0)
        if is_print: 
            printmd("$log(N_{ij})(>0)$")
            print(f"{log_conditionals}")
            print('-'*100)
        
        # Compute the log-likelihoods
        log_likelihoods -= log_conditionals
        log_likelihoods *= counts
        if is_print: 
            printmd("log_likelihoods = $N_{ijk} * log(N_{ijk}/N_{ij})$ = $N_{ijk}*[log(N_{ijk}) - log(N_{ij})]$ = ")
            print(f"{log_likelihoods}")
            print('-'*100)
        
        score = np.sum(log_likelihoods)
        score -= 0.5 * log(sample_size) * num_parents_states * (var_cardinality - 1)
        if is_print: 
            printmd("bic score = $\sum log\_likelihoods - 0.5*log(sample\_size(N)) * num_parents\_states(q_i)*(var\_cardinality(r_i) - 1)$")
            print(f"{log_likelihoods}")
            print('-'*100)
        
        
        return score
    
    def score(self, model,is_print=False):
        score = 0
        for node in model.nodes():
            if is_print:
                printmd(f"## node: {node}")
                printmd(f"### parent: {list(model.predecessors(node))}")
                printmd(f"score: {self.Bic_score_single(node, model.predecessors(node),is_print=is_print)}")
                print(f"="*100)
            score += self.Bic_score_single(node, model.predecessors(node))
        return score
    
    
class AicScore(BaseEstimator):
    def __init__(self, data, **kwargs):
        super(AicScore, self).__init__(data, **kwargs)
        
    def Aic_score_single(self, variable, parents, is_print=False):
        var_states = self.state_names[variable]
        var_cardinality = len(var_states) # number of element in a (undefined set) axiomatic set theory
        state_counts = self.state_counts(variable, parents)
        num_parents_states = float(state_counts.shape[1])
        counts = np.asarray(state_counts)
        sample_size = len(self.data)
        
        if is_print:
            printmd(f'r_i = :{var_cardinality}')
            printmd(f'q_i = :{num_parents_states}')
            printmd('state_counts($N_{ijk}$):'+str(counts))
            printmd('sample_size($N$):'+str(sample_size))
            display(state_counts.head(2))
            print('-'*100)
            
        log_likelihoods = np.zeros_like(counts, dtype=np.float_)
        
        # Compute the log-counts
        np.log(counts, out=log_likelihoods, where=counts > 0)
        if is_print: 
            printmd("$log(N_{ijk})(>0)$ = ")
            print(f"{log_likelihoods}")
            print('-'*100)
        
        # Compute the log-conditional sample size
        log_conditionals = np.sum(counts, axis=0, dtype=np.float_)
        np.log(log_conditionals, out=log_conditionals, where=log_conditionals > 0)
        if is_print: 
            printmd("$log(N_{ij})(>0)$")
            print(f"{log_conditionals}")
            print('-'*100)
        
        # Compute the log-likelihoods
        log_likelihoods -= log_conditionals
        log_likelihoods *= counts
        if is_print: 
            printmd("log_likelihoods = $N_{ijk} * log(N_{ijk}/N_{ij})$ = $N_{ijk}*[log(N_{ijk}) - log(N_{ij})]$ = ")
            print(f"{log_likelihoods}")
            print('-'*100)
        
        score = np.sum(log_likelihoods)
        score -= num_parents_states * (var_cardinality - 1)
        if is_print: 
            printmd("bic score = $\sum log\_likelihoods -  num_parents\_states(q_i)*(var\_cardinality(r_i) - 1)$")
            print(f"{log_likelihoods}")
            print('-'*100)
        
        
        return score
    
    def score(self, model,is_print=False):
        score = 0
        for node in model.nodes():
            if is_print:
                printmd(f"## node: {node}")
                printmd(f"### parent: {list(model.predecessors(node))}")
                printmd(f"score: {self.Aic_score_single(node, model.predecessors(node),is_print=is_print)}")
                print(f"="*100)
            score += self.Aic_score_single(node, model.predecessors(node))
        return score

    
    
class fNMLScore(BaseEstimator):
    def __init__(self, data, **kwargs):
        super(fNMLScore, self).__init__(data, **kwargs)
    
    def Szpankowski_approximation(self, row):
        if row.n == 0: return 1
        return 0.5*row.n*np.pi*np.exp(np.sqrt(8/(9*row.n*np.pi))+((3*np.pi-16)/(36*row.n*np.pi)))
    
    def Compute_regret(self, row, r):
        if row.n == 0: return 1
        return  row[f'C_{r-1}'] + (row.n/(r-2))*row[f'C_{r-2}']
        
    def fNML_score_single(self, variable, parents, is_print=False):
        var_states = self.state_names[variable]
        var_cardinality = len(var_states) # number of element in a (undefined set) axiomatic set theory
        state_counts = self.state_counts(variable, parents)
        num_parents_states = float(state_counts.shape[1])
        counts = np.asarray(state_counts)
        sample_size = len(self.data)
        
        if is_print:
            printmd(f'r_i = :{var_cardinality}')
            printmd(f'q_i = :{num_parents_states}')
            printmd('state_counts($N_{ijk}$):<br/>'+str(counts))
            printmd('sample_size($N$):'+str(sample_size))
            display(state_counts.head(2))
            print('-'*100)
            
        log_likelihoods = np.zeros_like(counts, dtype=np.float_)
        
        # Compute the log-counts
        np.log(counts, out=log_likelihoods, where=counts > 0)
        if is_print: 
            printmd("$log(N_{ijk})(>0)$ = <br/>"+f"{log_likelihoods}")
            print('-'*100)
        
        # Compute the log-conditional sample size
        log_conditionals = np.sum(counts, axis=0, dtype=np.float_)
        np.log(log_conditionals, out=log_conditionals, where=log_conditionals > 0)
        if is_print: 
            printmd("$log(N_{ij})(>0)$<br/>"+f"{log_conditionals}")
            print('-'*100)
        
        # Compute the log-likelihoods
        log_likelihoods -= log_conditionals
        log_likelihoods *= counts
        if is_print: 
            printmd("log_likelihoods = $N_{ijk} * log(N_{ijk}/N_{ij})$ = $N_{ijk}*[log(N_{ijk}) - log(N_{ij})]$ = ")
            print(f"{log_likelihoods}")
            print('-'*100)
        
        # Compute the 𝐶𝑁(𝐵𝐺) table
        N_ij = np.sum(counts, axis=0, dtype=np.float_)
        
        N_ij_unique, N_ij_cnt = np.unique(N_ij, return_counts=True)
        N_ij_dist = dict(zip(N_ij_unique, N_ij_cnt))
        
        C_ini = [[n, 1] for n in N_ij_unique]
        C_lookup_table = pd.DataFrame(C_ini, columns = ['n', 'C_1'])
        
        if var_cardinality > 1:
            C_lookup_table['C_2'] = C_lookup_table.apply(self.Szpankowski_approximation,axis=1)
        if var_cardinality > 2:
            for r_i in range(3,var_cardinality+1):
                Compute_regret_par = partial(self.Compute_regret, r=r_i)
                C_lookup_table[f'C_{r_i}'] = C_lookup_table.apply(Compute_regret_par,axis=1)
                if r_i > 5: #only obtain the last 3 regret C for memory saving
                    C_lookup_table = C_lookup_table.drop(columns=[f'C_{r_i-3}'])
        #C_lookup_table['N_ij_cnt'] = N_ij_cnt
        C_lookup_table['fNML_regret'] = np.log(C_lookup_table[f'C_{var_cardinality}'])
        
        fNML_regret = C_lookup_table.fNML_regret.values
        
        if is_print: 
            print(f"N_ij distribution:\n {N_ij_dist}")
            display(C_lookup_table)
            print("fNML_regret = \n"+f"{fNML_regret}")
            print('-'*100)
        
        score = np.sum(log_likelihoods)
        score -= np.sum(fNML_regret)
        if is_print: 
            printmd("fNML score = $\sum log\_likelihoods -  \sum logC^{r_i}_{N_{ij}}$")
            print('-'*100)
        return score
    
    def score(self, model,is_print=False):
        score = 0
        for node in model.nodes():
            if is_print:
                printmd(f"## node: {node}")
                printmd(f"### parent: {list(model.predecessors(node))}")
                printmd(f"score: {self.fNML_score_single(node, model.predecessors(node),is_print=is_print)}")
                print(f"="*100)
            score += self.fNML_score_single(node, model.predecessors(node))
        return score