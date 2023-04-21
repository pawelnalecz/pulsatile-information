import pandas as pd
from utils.math_utils import input_entropy

from core.step_manager import AbstractStep


class Step(AbstractStep):

    step_name = 'EMr'

    required_parameters = ['target_position', 'theoretical_parameters']
    input_files = ['extracted_slices', 'blinks']
    output_files = {'empirical_measures': '.pkl.gz', }

    def perform(self, **kwargs):
        print('------ EXTRACTING EMPIRICAL MEASURES ------')

        target_position = kwargs['target_position']
        min_interval = kwargs['theoretical_parameters']['min']

        slices: pd.DataFrame = self.load_file('extracted_slices')

        mean_interval = 1 / (slices['target'] == target_position).mean()


        empirical_measures = pd.Series({
            'mean interval': mean_interval,
            'input entropy': input_entropy(min_interval, mean_interval - min_interval),
            'input entropy assuming independent': input_entropy(0, mean_interval),
        })

        self.save_file(empirical_measures, 'empirical_measures')
