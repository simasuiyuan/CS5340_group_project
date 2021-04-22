from Model.ModelInterface import ModelInterface
from Model.BNpkg import utils, projection

from Model.BNpkg.bntrain import bn_struct_training

import bnlearn as bn

NUM_OF_CLS = 4
PERIOD = 5

class BNModel(ModelInterface):
    def __init__(self):
        super().__init__()

    def fit(self, training_data, **kwargs):
        # Data preprocessing
        cfg_preprocess = {'num_of_cls': NUM_OF_CLS, 'period': PERIOD}
        dict_train = utils.preprocess(training_data, **cfg_preprocess)
        self.metadata = dict_train['metadata']
        training_data = dict_train['df']
        self.inference_data = training_data.iloc[[-1]]

        # Structure learning
        cfg_sl = {'func': bn_struct_training, 
                    'params': {'sl_data': training_data, 'method': 'hillclimbsearch-modified', 'scoring': 'bic'}}
        self.struct = cfg_sl['func'](**cfg_sl['params'])

        # Parameter learning
        cfg_pl = {'func': bn.parameter_learning.fit, 'params': {'model': self.struct, 'df': training_data}}
        self.model = cfg_pl['func'](**cfg_pl['params'])

    def project(self, projection_data, **kwargs):
        max_proj_len = min(PERIOD, len(projection_data))
        
        cfg_proj =  {'func': projection.project, 'params': {'data': self.inference_data, 
                                        'metadata': self.metadata, 'model': self.model['model'], 'mode': 'simulated'}}
        return cfg_proj['func'](**cfg_proj['params'])[:max_proj_len]