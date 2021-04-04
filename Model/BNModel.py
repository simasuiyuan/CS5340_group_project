import BN_Inference
from ModelInterface import ModelInterface
import pandas as pd
import sys
sys.path.append('../')
from Xuhui_limin import training
class BNModel(ModelInterface):
    def __init__(self):
        super().__init__()
        self.variables_to_predict = []
        self.model = None

    def fit(self, training_data: pd.DataFrame, **kwargs):
        #pls add the name of the variable(s) to predict to the list
        # df = training.data_preparing()
        # df = training.GMM_clustering(5, 10, df)
        self.model = training.bn_model_training(training_data,'hc','bic')

    def project(self, projection_data: pd.DataFrame, **kwargs):
        predictions = infer_with_model(self.model, projection_data, self.variables_to_predict, model_type='bn_learn', output_type='dist')
        

    def summary(self):
        return self.model.summary()

if __name__ == '__main__':
    bn = BNModel()
    df = training.data_preparing()
    df = training.GMM_clustering(5, 10, df)
    bn.fit(df)
