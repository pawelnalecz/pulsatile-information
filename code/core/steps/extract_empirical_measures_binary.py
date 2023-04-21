import pandas as pd

from core.step_manager import AbstractStep

class Step(AbstractStep):

    step_name = 'EMb'

    required_parameters = []
    input_files = ['extracted_slices', 'blinks']
    output_files = {'empirical_measures': '.pkl.gz', }

    def perform(self, **kwargs):
        print('------ EXTRACTING EMPIRICAL MEASURES ------')

        slices: pd.DataFrame = self.load_file('extracted_slices')

        empirical_measures = pd.Series({
            'mean interval': 1 / slices['target'].mean()
        })

        self.save_file(empirical_measures, 'empirical_measures')
