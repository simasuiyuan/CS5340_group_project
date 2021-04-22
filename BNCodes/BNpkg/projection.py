import copy

import numpy as np
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel

np.random.seed(0)

def _infer(model, evidences, variables_to_predict):
    assert isinstance(model, BayesianModel)
    return VariableElimination(model).query(variables_to_predict, evidences, joint=False)   # Get marginal probabilities

def project(data, metadata, model, mode='simple'):

    def _extract_node_feature_data(data_keys, data_values):
        # Extract only the features that are present in the graph as nodes
        keys_ = []; values_ = []
        for col, v in zip(data_keys, data_values):
            if col in model.nodes:
                keys_.append(col)
                values_.append(v)
        return keys_, values_

    def _predict_root(ls_k, ls_v, mode):
        if mode == 'sequential':   # Sequential prediction with preceding available evidences
            evidences ={}
            for k, v in zip(ls_k, ls_v):
                if not (v is None):
                    evidences[k] = v

        elif mode == 'recursive':   # Recursive prediction with full evidences (using previously predicted values) at each timestep
            evidences = dict(zip(ls_k[1:], ls_v[1:]))

        else:
            print('UNKNOWN MODE')

        prob_dist = _infer(model, evidences, [data_keys[0]])[data_keys[0]].values
        return prob_dist
        

    res = []
    N = 10   # Number of samples per period
    num_of_cls = metadata['num_of_cls']
    p = metadata['period']   # Lookback
    arr_mean = np.array([m for _, m in metadata['mean'].items()])

    dm = metadata['discretization_model']

    data_copy = data.copy(deep=True)

    data_keys = data.keys()

    data_values = data.to_numpy()[0]
    data_values_copy = data_copy.to_numpy()[0]

    for i in range(p):
        # TODO: consider multiple time-series features if time permits

        # Shift all values back and set None on node to be predicted
        data_values = np.roll(data_values, 1)
        data_values[0] = None
        data_values_copy = np.roll(data_values_copy, 1)
        data_values_copy[0] = None

        ks_seq, vs_seq = _extract_node_feature_data(data_keys, data_values_copy)
        prob_dist_seq = _predict_root(ks_seq, vs_seq, 'sequential')

        ks_rec, vs_rec = _extract_node_feature_data(data_keys, data_values)
        prob_dist_rec = _predict_root(ks_rec, vs_rec, 'recursive')
        
        prob_dist = prob_dist_seq * prob_dist_rec
        prob_dist /= sum(prob_dist)

        idx = np.argmax(prob_dist)
        data_values[0] = idx

        if len(prob_dist) < num_of_cls:
            prob_dist = np.pad(prob_dist, (0, num_of_cls-len(prob_dist)))

        if mode == 'simple':
            res.append(arr_mean[idx])

        elif mode == 'weighted':

            scaled_val = sum(prob_dist*arr_mean)
            res.append(scaled_val)

        elif mode == 'simulated':
            
            dm.set_params(random_state=i+1)
            old_weights = copy.deepcopy(dm.weights_)
            dm.weights_ = prob_dist
            res.append(sum(dm.sample(N)[0].flatten())/N)
            dm.weights_ = old_weights

    print(res)
    return res
