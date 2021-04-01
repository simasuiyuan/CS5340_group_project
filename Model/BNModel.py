import BN_Inference

class BNModel(ModelInterface):
    def __init__(self):
        super().__init__()
        

    def fit(self, training_data: pd.DataFrame, **kwargs):
        self.model.fit(training_data)

    def project(self, projection_data: pd.DataFrame, **kwargs):
     

    def summary(self):
        return self.model.summary()
