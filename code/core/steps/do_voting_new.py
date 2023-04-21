import pandas as pd

from core.step_manager import AbstractStep
from utils.data_utils import index_without

from typing import Iterable


def prod(iterable: Iterable, start=None):
    retval = start if start is not None else 1
    for val in iterable:
        retval *= val
    return retval

class Step(AbstractStep):

    step_name = 'Vnew'

    required_parameters = ['target_position', 'voting_range', 'voting_threshold', 'correct_consecutive']
    input_files = ['prediction_results']
    output_files = {'binary_predictions': '.pkl.gz'}


    def perform(self, **kwargs):
        print('------ GENERATING BINARY PREDICTIONS ------')

        target_position = kwargs['target_position']
        voting_range = kwargs['voting_range']
        voting_threshold = kwargs['voting_threshold']
        correct_consecutive = kwargs['correct_consecutive']

        results: pd.DataFrame = self.load_file('prediction_results')
        
        pulse_no = results['pulse_no']

        binary_predictions = 1*pd.concat(
                sum(trajectory.shift(shift).eq(target_position - shift) for shift in voting_range) >= voting_threshold
                for ind, trajectory in results[['y_true', 'y_pred']].groupby(index_without(results, ['slice_no']))
            )
            
        if correct_consecutive:
            binary_predictions = pd.concat(
                trajectory * prod((1-trajectory.shift(i).fillna(0)) for i in range(1,correct_consecutive+1))  # removing the later of any two subsequent detections
                for ind, trajectory in binary_predictions.groupby(index_without(results, ['slice_no']))
            )
        
        binary_predictions = binary_predictions.join(pulse_no).astype('int64')
        
        # change slice_no to time_point by shifting by target_position
        old_index = index_without(binary_predictions, ['slice_no'])
        binary_predictions = binary_predictions.reset_index().assign(time_point=lambda x: x['slice_no'] - target_position).drop(columns=['slice_no']).set_index(old_index + ['time_point'])

        self.save_file(binary_predictions, 'binary_predictions')


