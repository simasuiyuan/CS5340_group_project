import BN_Inference

class BNModel(ModelInterface):
    def __init__(self):
        super().__init__()
        self.variables_to_predict = []
        self.model = None

    def fit(self, training_data: pd.DataFrame, **kwargs):
        #pls add the name of the variable(s) to predict to the list
        self.model.fit(training_data)

    def project(self, projection_data: pd.DataFrame, **kwargs):
        predictions = infer_with_model(self.model, projection_data, self.variables_to_predict, model_type='bn_learn', output_type='dist')
        

    def summary(self):
        return self.model.summary()
