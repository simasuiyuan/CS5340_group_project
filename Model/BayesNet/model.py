class BNModel:
    """
    Args:
        config_sl (dict): dict of dict containing the function and its parameters for structure learning. 
        E.g. {'func': bn_struct_training, 'params': {'sl_data': SL_TRAIN, 'method': 'hillclimbsearch-modified', 'scoring': 'bic'}}
    """

    def __init__(self, config_sl):
        self.struct = config_sl['func'](**config_sl['params'])

    def fit(self, config_pl):
        """
        Args:
            config_pl (dict): dict of dict containing the function and its parameters for parameter learning.
            E.g. {'func': bn.parameter_learning.fit, 'params': {'model': model, 'df': PL_TRAIN,}}

        """
        self.model = config_pl['func'](**config_pl['params'])

    def project(self, config_proj):
        """
        Args:
            config_pl (dict): dict of dict containing the function and its parameters for projection.
            E.g. {'func': _project, 'params': {'data': PROJ_TEST, 'metadata': metadata, 'model': model, 'mode': 'simple'}}

        """
        return config_proj['func'](**config_proj['params'])