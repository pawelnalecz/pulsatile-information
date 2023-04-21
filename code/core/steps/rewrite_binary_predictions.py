import pandas as pd

from core.step_manager import AbstractStep
from utils.data_utils import index_without


class Step(AbstractStep):

    step_name = 'RBP'

    required_parameters = ['correct_consecutive', 'target_position']
    input_files = ['prediction_results']
    output_files = {'binary_predictions': '.pkl.gz'}


    def perform(self, **kwargs):
        print('------ REWRITING BINARY PREDICTIONS ------')

        target_position = kwargs['target_position']
        correct_consecutive = kwargs['correct_consecutive']

        results: pd.DataFrame = self.load_file('prediction_results')
        
        pulse_no = results['pulse_no']

        binary_predictions = results[['y_true', 'y_pred']]

        if correct_consecutive:
            binary_predictions = pd.concat(
                trajectory - (trajectory * trajectory.shift(1)).fillna(0)  # removing the later of any two subsequent detections
                for ind, trajectory in binary_predictions.groupby(index_without(binary_predictions, ['slice_no']))
            )
        
        binary_predictions = binary_predictions.join(pulse_no).astype('int64')
        
        # change slice_no to time_point by shifting by target_position
        old_index = index_without(binary_predictions, ['slice_no'])
        binary_predictions = binary_predictions.reset_index().assign(time_point=lambda x: x['slice_no'] - target_position).drop(columns=['slice_no']).set_index(old_index + ['time_point'])

        self.save_file(binary_predictions, 'binary_predictions')


