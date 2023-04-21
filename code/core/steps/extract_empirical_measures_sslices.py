import pandas as pd
from utils.math_utils import plogp, input_entropy


from core.step_manager import AbstractStep

class Step(AbstractStep):

    step_name = 'EMss'

    required_parameters = ['s_slice_length', 'theoretical_parameters']
    input_files = ['extracted_slices']
    output_files = {'empirical_measures': '.pkl.gz', }

    def perform(self, **kwargs):
        print('------ EXTRACTING EMPIRICAL MEASURES ------')

        s_slice_length = kwargs['s_slice_length']
        min_interval = kwargs['theoretical_parameters']['min']

        slices: pd.DataFrame = self.load_file('extracted_slices')

        mean_interval = 1 / (slices['target'] % 2).mean()

        input_entropy_assuming_independent = (plogp(slices.groupby('target').size()).sum() - plogp(len(slices))) / len(slices) / s_slice_length

        empirical_measures = pd.Series({
            'mean interval': mean_interval,
            'input entropy': input_entropy(min_interval, mean_interval - min_interval),
            'input entropy assuming independent': input_entropy_assuming_independent,
        })

        self.save_file(empirical_measures, 'empirical_measures')
