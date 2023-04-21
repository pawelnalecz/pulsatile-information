import pandas as pd
from utils.math_utils import plogp, input_entropy


from core.step_manager import AbstractStep

class Step(AbstractStep):

    step_name = 'EMap'

    required_parameters = ['s_slice_length', 'theoretical_parameters']
    input_files = ['binary_predictions']
    output_files = {'empirical_measures': '.pkl.gz', }

    def perform(self, **kwargs):
        print('------ EXTRACTING EMPIRICAL MEASURES ------')

        s_slice_length = kwargs['s_slice_length']
        min_interval = kwargs['theoretical_parameters']['min']

        binary_predictions: pd.DataFrame = self.load_file('binary_predictions')

        mean_interval = 1 / (binary_predictions['y_true'] % 2).mean()

        input_distribition = binary_predictions.groupby('y_true').size()
        total_count = input_distribition.sum()
        input_entropy_assuming_independent = (plogp(input_distribition).sum() - plogp(total_count)) / total_count / s_slice_length

        empirical_measures = pd.Series({
            'mean interval': mean_interval,
            'input entropy': input_entropy(min_interval, mean_interval - min_interval),
            'input entropy assuming independent': input_entropy_assuming_independent,
        })

        self.save_file(empirical_measures, 'empirical_measures')
