# learning_on_good.py

import numpy as np
import pandas as pd

from utils.utils import list_without
from core.step_manager import AbstractStep, Chain

class Step(AbstractStep):

    step_name = 'BT'

    required_parameters = [
        'slice_length',
        'target_position', 
        'pulse_window_matching_shift',
        'remove_first_pulses', 
        'remove_break', 
        'remove_shorter_than', 
        'correct_consecutive',
        'binary_timeline_threshold', 
        'yesno', 
        'timeline_extraction_method', 
    ]
    input_files = ['prediction_results']
    output_files = {'binary_timeline': '.pkl.gz'}

    def __init__(self, chain: Chain) -> None:
        if chain.parameters['remove_break']:
            self.input_files = self.input_files + ['previous_pulse_lengths', 'blinks']
        super().__init__(chain)



    def perform(self, **kwargs):

        slice_length = kwargs['slice_length']
        target_position = kwargs['target_position']
        pulse_window_matching_shift = kwargs['pulse_window_matching_shift']
        remove_first_pulses = kwargs['remove_first_pulses']
        remove_break = kwargs['remove_break']
        remove_shorter_than = kwargs['remove_shorter_than']
        correct_consecutive = kwargs['correct_consecutive']
        binary_timeline_threshold = kwargs['binary_timeline_threshold']
        timeline_extraction_method = kwargs['timeline_extraction_method']
        yesno = kwargs['yesno']

        print('------ EXTRACTING RESULT TIMELINE ------')

        results : pd.DataFrame = self.load_file('prediction_results')
        
        if remove_first_pulses:
            results  = results[results['pulse_no'] >= remove_first_pulses]
        if remove_break:            
            previous_pulse_lengths : pd.Series = self.load_file('previous_pulse_lengths')
            blinks : pd.Series = self.load_file('blinks')        
            this_pulse_lengths = previous_pulse_lengths.shift(-1).rename('this_pulse_length')
            next_pulse_lengths = previous_pulse_lengths.shift(-2).rename('next_pulse_length')
            results = results[(lambda x: (x['previous_pulse_length'] <remove_break) & ((x['this_pulse_length'] < remove_break) |  (x.index.get_level_values('slice_no') - x['signal_on_timepoint_index'] +slice_length - pulse_window_matching_shift -1 < x['next_pulse_length'])))(results.join(previous_pulse_lengths, on='pulse_no').join(this_pulse_lengths, on='pulse_no').join(next_pulse_lengths, on='pulse_no').join(blinks, on='pulse_no'))] #filter out the break in the middle
        if remove_shorter_than:
            long_enough_tracks = (lambda x: x[x>=remove_shorter_than])(results.groupby(['classifier', 'iteration', 'track_id']).size()).index.get_level_values('track_id')
            results = results[results.index.get_level_values('track_id').isin(long_enough_tracks)]


        if 'subsequent' in timeline_extraction_method:
            time_step = (lambda x: x[1]-x[0])(np.sort(results['y_true'].unique()))

        methods = {
            'normal': lambda : 1*results['y_pred'].eq(target_position), 
            'subsequent': lambda : 1*(
                  1*results.groupby(list_without(results.index.names, 'slice_no')).shift(1)['y_pred'].eq(target_position-time_step) 
                + 1*results['y_pred'].eq(target_position) 
                + 1*results.groupby(list_without(results.index.names, 'slice_no')).shift(-1) ['y_pred'].eq(target_position+time_step)
                 >= 2),
            'subsequent_3_of_5': lambda : 1*(
                  1*results.groupby(list_without(results.index.names, 'slice_no')).shift(2)['y_pred'].eq(target_position-2*time_step) 
                + 1*results.groupby(list_without(results.index.names, 'slice_no')).shift(1)['y_pred'].eq(target_position-time_step) 
                + 1*results['y_pred'].eq(target_position) 
                + 1*results.groupby(list_without(results.index.names, 'slice_no')).shift(-1) ['y_pred'].eq(target_position+time_step)
                + 1*results.groupby(list_without(results.index.names, 'slice_no')).shift(-2) ['y_pred'].eq(target_position+2*time_step)
                 >= 3), 
            'subsequent_2_of_5': lambda : 1*(
                  1*results.groupby(list_without(results.index.names, 'slice_no')).shift(2)['y_pred'].eq(target_position-2*time_step) 
                + 1*results.groupby(list_without(results.index.names, 'slice_no')).shift(1)['y_pred'].eq(target_position-time_step) 
                + 1*results['y_pred'].eq(target_position) 
                + 1*results.groupby(list_without(results.index.names, 'slice_no')).shift(-1) ['y_pred'].eq(target_position+time_step)
                + 1*results.groupby(list_without(results.index.names, 'slice_no')).shift(-2) ['y_pred'].eq(target_position+2*time_step)
                 >= 2), 
            }

        print('computing binary timeline', end=' ...', flush=True)
        binary_results = 1*results[['y_true', 'y_pred']].rename(columns={'y_true':'input_blinks', 'y_pred': 'output_detections'}) if yesno else pd.DataFrame({
            'input_blinks': 1*results['y_true'].eq(target_position), 
            'output_detections': methods[timeline_extraction_method](), 
            })


        if correct_consecutive:
            print('correcting consecutive', end=' ...', flush=True)
            #binary_results  = binary_results * (1- (binary_results.groupby(list_without(binary_results.index.names, 'slice_no')).shift(1, fill_value=0) + binary_results.groupby(list_without(binary_results.index.names, 'slice_no')).shift(-1, fill_value=0))/2)
            for i in range(correct_consecutive):
                binary_results  = binary_results * (1- binary_results.groupby(list_without(binary_results.index.names, 'slice_no')).shift(i+1, fill_value=0))

            # binary_results  = pd.concat([
            #     track - 0.5 * (track & (track.shift(-1, fill_value=0) | track.shift(1, fill_value=0)))
            #     for ind, track in binary_results.groupby(list_without(binary_results.index.names, 'slice_no'))
            # ])
        


        print('averaging', end=' ...', flush=True)

        binary_timeline = (1*binary_results).groupby(list_without(binary_results.index.names, 'repetition')).mean().sort_index()
        if binary_timeline_threshold: 
            binary_timeline = 1*(binary_timeline >= binary_timeline_threshold)       
            if correct_consecutive:
                print('correcting consecutive', end=' ...', flush=True)
                for i in range(correct_consecutive):
                    binary_timeline  = binary_timeline * (1- binary_timeline.groupby(list_without(binary_timeline.index.names, 'slice_no')).shift(i+1, fill_value=0))



        print('done.', flush=True)

        self.save_file(binary_timeline, 'binary_timeline')

