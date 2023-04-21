import pandas as pd

from core.step_manager import AbstractStep, Chain
from utils.data_utils import index_without


class Step(AbstractStep):

    step_name = 'Vb'

    required_parameters = ['voting_range', 'voting_threshold', 'correct_consecutive']
    input_files = []
    output_files = {'binary_predictions': '.pkl.gz'}

    def __init__(self, chain: Chain) -> None:
        for shift in chain.parameters['voting_range']:
            self.input_files = self.input_files + [f"binary_predictions_{shift:d}"]
        super().__init__(chain)


    def perform(self, **kwargs):
        print('------ VOTING (BOOST) ------')

        voting_range = kwargs['voting_range']
        voting_threshold = kwargs['voting_threshold']
        correct_consecutive = kwargs['correct_consecutive']

        binary_predictions: pd.DataFrame = sum([self.load_file(f"binary_predictions_{shift:d}")[['y_true', 'y_pred']] for shift in voting_range]).dropna() # type: ignore

        binary_predictions = 1*(binary_predictions >= voting_threshold)
        
        if correct_consecutive:
            binary_predictions = pd.concat(
                trajectory - (trajectory * trajectory.shift(1)).fillna(0)  # removing the later of any two subsequent detections
                for ind, trajectory in binary_predictions.groupby(index_without(binary_predictions, ['slice_no']))
            )

        pulse_no = self.load_file(f"binary_predictions_{voting_range[0]:d}")['pulse_no']
        binary_predictions = binary_predictions.join(pulse_no).astype('int64')
        
        self.save_file(binary_predictions, 'binary_predictions')


