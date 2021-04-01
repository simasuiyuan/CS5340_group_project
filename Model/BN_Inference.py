from pgmpy.inference import VariableElimination
import pandas as pd
import bnlearn
import numpy as np
'''
Inputs:
    1. model: 
    The trained BN model with CPD filled up.
    The variable to predict is included in the model.
    
    2. evidence_data: 
    The discretized evidence data in the form of panda data frame.
    Each row is a data point.
    The columns are the evidence variables. 
    
    3. to_predict
    The list of variables to predict as a list of strings.
    
    4. model_type
    'pgypy' or 'bnlearn'
    
    5. output_type
    'dist' or 'map'
    'dist' to return the posterior distribution
    'map' to return a list of dictionaries mapping the variables to the MAP values.
    
    
Output:
    predictions:
        Depends on output_type:
        'dist' or 'map'
        'dist' to return the posterior distribution
        'map' to return a list of dictionaries mapping the variables to the MAP values.

'''
def infer_with_model(model, evidence_data, to_predict, model_type='pgmpy', output_type='map'):
    predictions = []
    evidences = evidence_data.to_dict(orient='records')
    if model_type=='pgmpy':
        infer = VariableElimination(model)
        for evidence in evidences:
            if output_type=='map':
                predictions.append(infer.map_query(to_predict, evidence))
            elif output_type=='dist':
                predictions.append(infer.query(to_predict, evidence))
            else:
                print('Unknown model type')
    elif model_type=='bnlearn':
        for evidence in evidences:
            if output_type=='map':
                prediction = bnlearn.inference.fit(model, to_predict, evidence)
                values = prediction.values
                variables = prediction.variables
                indices = np.unravel_index(values.argmax(), values.shape)
                d = dict()
                for i in range(len(variables)):
                    variable = variables[i]
                    d[variable] = indices[i]
                predictions.append(d)
            elif output_type=='dist':
                predictions.append(bnlearn.inference.fit(model, to_predict, evidence))
            else:
                print('Unknown model type')
    return predictions
    
'''
Inputs:
    1. predictions:
    A list of dictionaries mapping predicted variables to predicted categorical value.
    
    2.     
Output:
    1. projections:
    
'''

def categorical_prediction_to_projection(predictions):
    return projections