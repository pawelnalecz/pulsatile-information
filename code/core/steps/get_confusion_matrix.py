from core.step_manager import AbstractStep, printor
from utils.math_utils import rolling_binary_sequence_to_int
from utils.data_utils import index_without
import pandas as pd


class Step(AbstractStep):

    step_name = 'CM'

    required_parameters = ['s_slice_length', 'r_slice_length', 'fields_reduced_with_confusion_matrix']
    input_files = ['binary_predictions']
    output_files = {'confusion_matrix': '.pkl.gz'}



    def perform(self, **kwargs):
        print('------ COMPUTING CONFUSION MATRIX ------')

        s_slice_length = kwargs['s_slice_length']
        r_slice_length = kwargs['r_slice_length']
        fields_reduced_with_confusion_matrix = kwargs['fields_reduced_with_confusion_matrix']

        binary_predictions: pd.DataFrame = self.load_file('binary_predictions')
    
        print('Converting rolling binary slices to int and computing the confusion matrix...', flush=True)
        sliced_predictions = pd.concat({
                ind: trajectory.agg({
                    'y_true': lambda x: rolling_binary_sequence_to_int(x, s_slice_length),
                    'y_pred': lambda x: rolling_binary_sequence_to_int(x, r_slice_length)
                    }).dropna().astype(int)# rolling_binary_sequence_to_int(trajectory, r_slice_length).dropna().astype('int').groupby(['y_true', 'y_pred']).size()
                for ind, trajectory in binary_predictions[['y_true', 'y_pred']].dropna().groupby(index_without(binary_predictions, ['time_point']))
            }, names=index_without(binary_predictions, ['time_point']))
        confusion_matrix = sliced_predictions.groupby(index_without(binary_predictions, fields_reduced_with_confusion_matrix) + ['y_true', 'y_pred']).size()
        print('done.')
        
        self.save_file(confusion_matrix, 'confusion_matrix')


