import pandas as pd
import numpy as np

from core.step_manager import AbstractStep

from utils.math_utils import plogp, input_entropy, conditional_entropy



def loss_due_to_FP(x):
    return plogp(x['TP'])+plogp(x['FP'])- plogp(x['TP']+x['FP']) + (plogp(x['TP'] + x['FN']) - plogp(x['TP'] + x['FN'] + x['FP']) - plogp(x['TP']) + plogp(x['TP'] + x['FP']))

def loss_due_to_FN(x):
    return plogp(x['FN'])+plogp(x['TN'])- plogp(x['FN']+x['TN']) - (plogp(x['TP'] + x['FN']) - plogp(x['TP'] + x['FN'] + x['FP']) - plogp(x['TP']) + plogp(x['TP'] + x['FP']))


class Step(AbstractStep):

    step_name = 'IT'

    required_parameters = ['theoretical_parameters', 'loss_source_determination_method']
    input_files = ['binary_timeline']
    output_files = {'information_overall_empirical': '.pkl.gz', 'information_overall_theoretical': '.pkl.gz', 'information_per_track_empirical': '.pkl.gz', 'information_per_track_theoretical': '.pkl.gz'}


    def perform(self, **kwargs):
        theoretical_min = kwargs['theoretical_parameters']['min']
        theoretical_exp_mean = kwargs['theoretical_parameters']['exp_mean']
        minutes_per_timepoint = kwargs['theoretical_parameters']['minutes_per_timepoint']
        method = kwargs['loss_source_determination_method']

        print('------------ COMPUTING INFORMATION TRANSMISSION RATE ------------')
        isSequential = method[:21] == 'sequential_correction' or method == 'sequential_averaged'
        isFixing = method[:5] == 'fixing'
        assert method in ['full_matrix'] or isSequential or isFixing

        if method == 'full_matrix':
            def loss_due_to_FP(x):
                return plogp(x['TP'])+plogp(x['FP'])- plogp(x['TP']+x['FP'])

            def loss_due_to_FN(x):
                return plogp(x['FN'])+plogp(x['TN'])- plogp(x['FN']+x['TN']) #(plogp(x['TP'] + x['FN']) - plogp(x['TP'] + x['FN'] + x['FP']) - plogp(x['TP']) + plogp(x['TP'] + x['FP']))

        if isFixing or isSequential:

            if method == 'sequential_averaged':
                def loss_due_to_FP(x):
                    return 1/2*(plogp(x['FP']) +  plogp(x['TP'] + x['FN']) - plogp(x['TP'] + x['FN'] + x['FP']) + plogp(x['FP']) + plogp(x['TN']) + plogp(x['TP']) - plogp(x['FP'] + x['TP']) - plogp(x['FN'] + x['TN']) - plogp(x['TN'] + x['FP']) +plogp(x['TN'] + x['FP'] + x['FN']))

                def loss_due_to_FN(x):
                    return 1/2*(plogp(x['FN']) +  plogp(x['TN'] + x['FP']) - plogp(x['TN'] + x['FP'] + x['FN']) + plogp(x['FN']) + plogp(x['TP']) + plogp(x['TN']) - plogp(x['FN'] + x['TN']) - plogp(x['FP'] + x['TP']) - plogp(x['TP'] + x['FN']) +plogp(x['TP'] + x['FN'] + x['FP']))
                

            else:
                error_correction_sequence = method[(22 if isSequential else 7):].split('_')
                assert len(error_correction_sequence) == 2 and all(kind_of_error in error_correction_sequence for kind_of_error in ('FN', 'FP'))
                for kind_of_error in error_correction_sequence:
                    if kind_of_error == 'FN':
                        FN_first = True
                        break
                    if kind_of_error == 'FP':
                        FN_first = False
                        break

                if FN_first:
                    
                    def loss_due_to_FP(x):
                        return plogp(x['FP']) +  plogp(x['TP'] + x['FN']) - plogp(x['TP'] + x['FN'] + x['FP'])

                    def loss_due_to_FN(x):
                        return plogp(x['FN']) + plogp(x['TP']) + plogp(x['TN']) - plogp(x['FN'] + x['TN']) - plogp(x['FP'] + x['TP']) - plogp(x['TP'] + x['FN']) +plogp(x['TP'] + x['FN'] + x['FP'])
                else:
                    def loss_due_to_FP(x):
                        return plogp(x['FP']) + plogp(x['TN']) + plogp(x['TP']) - plogp(x['FP'] + x['TP']) - plogp(x['FN'] + x['TN']) - plogp(x['TN'] + x['FP']) +plogp(x['TN'] + x['FP'] + x['FN'])

                    def loss_due_to_FN(x):
                        return plogp(x['FN']) +  plogp(x['TN'] + x['FP']) - plogp(x['TN'] + x['FP'] + x['FN'])


        binary_timeline_with_correction_for_consecutive : pd.DataFrame = self.load_file('binary_timeline')

        theoretical_mean = theoretical_min + theoretical_exp_mean
        
        theoretical_input_frequency : float= 1/theoretical_mean# if not regular else 0.5
        theoretical_input_entropy : float = input_entropy(theoretical_min, theoretical_exp_mean)# if not regular else 1
        theoretical_input_entropy_assuming_poisson : float = input_entropy(0, theoretical_mean)# if not regular else 1

        
        def norm_for_theoretical_input(x):
            dictionary = {
                    'TP': x['TP'] * theoretical_input_frequency / x['input_pulses'],
                    'FN': x['FN'] * theoretical_input_frequency / x['input_pulses'],
                    'FP': x['FP'] * (1-theoretical_input_frequency) / (1-x['input_pulses']),
                    'TN': x['TN'] * (1-theoretical_input_frequency) / (1-x['input_pulses']),
                    'input_pulses': x['input_pulses'] * theoretical_input_frequency / x['input_pulses'],
                    'output_pulses': x['TP'] * theoretical_input_frequency / x['input_pulses'] + x['FP'] * (1-theoretical_input_frequency) / (1-x['input_pulses']),
                }
            if type(x) == pd.DataFrame: return pd.DataFrame(dictionary, index = x.index)
            else: return pd.Series(dictionary)
        

        print("Scoring", end='... ', flush=True)
        
        scores_with_correction_for_consecutive = pd.DataFrame({
            ind: {
                'track_length': len(track),
                'TP': (track['input_blinks'] * track['output_detections']).sum(),
                'FN': (track['input_blinks'] * (1-track['output_detections'])).sum(),
                'FP': ((1-track['input_blinks']) * track['output_detections']).sum(),
                'TN': ((1-track['input_blinks']) * (1-track['output_detections'])).sum(),
                'input_pulses': track['input_blinks'].sum(),
                'output_pulses': track['output_detections'].sum(),
            }
            for ind, track in binary_timeline_with_correction_for_consecutive.groupby(['classifier', 'iteration', 'track_id'])

        }).rename_axis(columns=['classifier', 'iteration', 'track_id']).T
        
        scores_with_correction_for_consecutive = scores_with_correction_for_consecutive[(scores_with_correction_for_consecutive['input_pulses'] > 0) & (scores_with_correction_for_consecutive['input_pulses'] < scores_with_correction_for_consecutive['track_length'] )]

        print("Normalizing", end='... ', flush=True)
        
        scores_with_correction_for_consecutive_per_minute = scores_with_correction_for_consecutive.div(scores_with_correction_for_consecutive['track_length'], axis=0).assign(
            input_entropy=lambda x: input_entropy(theoretical_min, 1/x['input_pulses'] - theoretical_min),
            conditional_entropy=conditional_entropy,
        )


        theoretical_scores_with_correction_for_consecutive_per_minute = norm_for_theoretical_input(scores_with_correction_for_consecutive_per_minute).assign(
            input_entropy=lambda x: theoretical_input_entropy,
            conditional_entropy=conditional_entropy,
        )
        print("Done", flush=True)

        overall_empirical : pd.Series = (lambda x: x.sum()/x['track_length'].sum())(scores_with_correction_for_consecutive)
        overall_empirical['input_entropy'] = input_entropy(theoretical_min, 1/overall_empirical['input_pulses'] - theoretical_min)
        overall_empirical['conditional_entropy'] = conditional_entropy(overall_empirical)
        overall_empirical['information lost due to false detections[b/timepoint]'] = loss_due_to_FP(overall_empirical)
        overall_empirical['information lost due to missed pulses[b/timepoint]'] = loss_due_to_FN(overall_empirical)
        overall_empirical['channel_capacity[b/timepoint]'] = overall_empirical['input_entropy'] - overall_empirical['conditional_entropy']
        overall_empirical['channel_capacity_assuming_poisson[b/timepoint]'] = input_entropy(0, 1/overall_empirical['input_pulses']) - overall_empirical['conditional_entropy']
        overall_empirical['channel_capacity[b/h]'] = 60/minutes_per_timepoint*overall_empirical['channel_capacity[b/timepoint]']
        overall_empirical['channel_capacity_assuming_poisson[b/h]'] = 60/minutes_per_timepoint*overall_empirical['channel_capacity_assuming_poisson[b/timepoint]']
        overall_empirical['channel_capacity[min/b]'] = minutes_per_timepoint/overall_empirical['channel_capacity[b/timepoint]']
        overall_empirical['channel_capacity_assuming_poisson[min/b]'] = minutes_per_timepoint/overall_empirical['channel_capacity_assuming_poisson[b/timepoint]']
        

        overall_theoretical : pd.Series  = norm_for_theoretical_input(overall_empirical)
        overall_theoretical['input_entropy'] = theoretical_input_entropy
        overall_theoretical['conditional_entropy'] = conditional_entropy(overall_theoretical)
        overall_theoretical['information lost due to false detections[b/timepoint]'] = loss_due_to_FP(overall_theoretical)
        overall_theoretical['information lost due to missed pulses[b/timepoint]'] = loss_due_to_FN(overall_theoretical)
        overall_theoretical['channel_capacity[b/timepoint]'] = overall_theoretical['input_entropy'] - overall_theoretical['conditional_entropy']
        overall_theoretical['channel_capacity_assuming_poisson[b/timepoint]'] = theoretical_input_entropy_assuming_poisson - overall_theoretical['conditional_entropy']
        overall_theoretical['channel_capacity[b/h]'] = 60/minutes_per_timepoint*overall_theoretical['channel_capacity[b/timepoint]']
        overall_theoretical['channel_capacity_assuming_poisson[b/h]'] = 60/minutes_per_timepoint*overall_theoretical['channel_capacity_assuming_poisson[b/timepoint]']
        overall_theoretical['channel_capacity[min/b]'] = minutes_per_timepoint/overall_theoretical['channel_capacity[b/timepoint]']
        overall_theoretical['channel_capacity_assuming_poisson[min/b]'] = minutes_per_timepoint/overall_theoretical['channel_capacity_assuming_poisson[b/timepoint]']
        
        per_track_empirical = scores_with_correction_for_consecutive_per_minute.groupby('track_id').mean()
        per_track_empirical['information lost due to false detections[b/timepoint]'] = loss_due_to_FP(per_track_empirical)
        per_track_empirical['information lost due to missed pulses[b/timepoint]'] = loss_due_to_FN(per_track_empirical)
        per_track_empirical['channel_capacity[b/timepoint]'] = input_entropy(theoretical_min, 1/per_track_empirical['input_pulses'] - theoretical_min)- per_track_empirical['conditional_entropy']
        per_track_empirical['channel_capacity_assuming_poisson[b/timepoint]'] = input_entropy(0, 1/per_track_empirical['input_pulses']) - per_track_empirical['conditional_entropy']
        per_track_empirical['channel_capacity[b/h]'] = 60/minutes_per_timepoint*per_track_empirical['channel_capacity[b/timepoint]']
        per_track_empirical['channel_capacity_assuming_poisson[b/h]'] = 60/minutes_per_timepoint*per_track_empirical['channel_capacity_assuming_poisson[b/timepoint]']
        per_track_empirical['channel_capacity[min/b]'] = minutes_per_timepoint/per_track_empirical['channel_capacity[b/timepoint]']
        per_track_empirical['channel_capacity_assuming_poisson[min/b]'] = minutes_per_timepoint/per_track_empirical['channel_capacity_assuming_poisson[b/timepoint]']
        
        per_track_theoretical = theoretical_scores_with_correction_for_consecutive_per_minute.groupby('track_id').mean()
        per_track_theoretical['information lost due to false detections[b/timepoint]'] = loss_due_to_FP(per_track_theoretical)
        per_track_theoretical['information lost due to missed pulses[b/timepoint]'] = loss_due_to_FN(per_track_theoretical)
        per_track_theoretical['channel_capacity[b/timepoint]'] = theoretical_input_entropy - per_track_theoretical['conditional_entropy']
        per_track_theoretical['channel_capacity_assuming_poisson[b/timepoint]'] = theoretical_input_entropy_assuming_poisson - per_track_theoretical['conditional_entropy']
        per_track_theoretical['channel_capacity[b/h]'] = 60/minutes_per_timepoint*per_track_theoretical['channel_capacity[b/timepoint]']
        per_track_theoretical['channel_capacity_assuming_poisson[b/h]'] = 60/minutes_per_timepoint*per_track_theoretical['channel_capacity_assuming_poisson[b/timepoint]']
        per_track_theoretical['channel_capacity[min/b]'] = minutes_per_timepoint/per_track_theoretical['channel_capacity[b/timepoint]']
        per_track_theoretical['channel_capacity_assuming_poisson[min/b]'] = minutes_per_timepoint/per_track_theoretical['channel_capacity_assuming_poisson[b/timepoint]']
        

        self.save_file(overall_empirical, 'information_overall_empirical')
        self.save_file(overall_theoretical, 'information_overall_theoretical')
        self.save_file(per_track_empirical, 'information_per_track_empirical')
        self.save_file(per_track_theoretical, 'information_per_track_theoretical')
