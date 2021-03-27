from pgmpy.inference import VariableElimination
import pandas as pd
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
    
Output:
    predictions:
    A list of dictionaries. Mapping the predicted variables to the predicted values for each row of the
    input evidence data.

'''
def infer_with_model(model, evidence_data, to_predict):
    predictions = []
    infer = VariableElimination(model)
    evidences = evidence_data.to_dict(orient='records')
    for evidence in evidences:
        predictions.append(infer.map_query(to_predict, evidence))
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