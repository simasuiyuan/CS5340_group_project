import numpy as np
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel


def _infer(model, evidences, variables_to_predict):
    assert isinstance(model, BayesianModel)
    return VariableElimination(model).query(variables_to_predict, evidences, joint=False)   # Get marginal probabilities

def project(data, metadata, model, mode='simple'):
    res = []
    p = metadata['period']
    m = np.array([m for _, m in metadata['mean'].items()])
    data_keys = data.keys()
    data_values = data.to_numpy()[0]
    for _ in range(p):
        # TODO: consider multiple time-series features if time permits
        data_values[0] = None
        
        ls_to_predict = []
        evidences ={}
        for k, v in zip(data_keys, data_values):
            if v is None:
                ls_to_predict.append(k)
            else:
                evidences[k] = v
        data_values = np.roll(data_values, 1)
        prob_dist = _infer(model, evidences, ls_to_predict)[data_keys[0]].values

        if mode == 'simple':
            idx = np.argmax(prob_dist)
            res.append(m[data_keys[idx]])
        elif mode == 'weighted':
            res.append(sum(prob_dist*m)*100)
    
    return res