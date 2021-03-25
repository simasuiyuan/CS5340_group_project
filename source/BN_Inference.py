
'''
Inputs:
    1. model: 
    The trained BN model with CPD filled up.
    The model contains all the observed variables and the variable to predict.
    2. observed_data: 
    The discretized observed data in the form of panda data frame.
    Each row is a data point.
    The columns are the observed variables. 
Output:
    prediction:
    The discrete category of the unobserved variable with all the other variables as evidence.

'''
def infer_with_model(model, observed_data):
    return prediction