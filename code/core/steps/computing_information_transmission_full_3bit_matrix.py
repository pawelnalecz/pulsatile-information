import pandas as pd
import numpy as np

from progressbar import ProgressBar
from itertools import combinations, permutations

from core.step_manager import AbstractStep
from utils.math_utils import plogp, input_entropy, bernoulli_entropy_rate


def correct_ID(timeline):
    
    out = timeline.copy()
    out['output_detections'] = out['output_detections'] + (1-out['output_detections'])*out['output_detections'].shift(1) * out['input_blinks'] - ((1-out['output_detections'])*out['output_detections'].shift(1) * out['input_blinks']).shift(-1)
    out['output_detections'] = out['output_detections'] + (1-out['output_detections'])*out['output_detections'].shift(-1) * out['input_blinks'] - ((1-out['output_detections'])*out['output_detections'].shift(-1) * out['input_blinks']).shift(1)

    return out


def correct_FN(timeline):
    
    out = timeline.copy()
    out['output_detections'] = out['output_detections'] +  out['input_blinks'] * (1 - out['output_detections'].shift(1)) * (1 - out['output_detections']) * (1 - out['output_detections'].shift(-1))

    return out

def correct_FP(timeline):
    
    out = timeline.copy()
    out['output_detections'] = out['output_detections'] -  out['output_detections'] * (1 - out['input_blinks'].shift(1)) * (1 - out['input_blinks']) * (1 - out['input_blinks'].shift(-1)) - out['output_detections'] * out['input_blinks'].shift(1) * (out['output_detections'].shift(1) + out['output_detections'].shift(2) - out['output_detections'].shift(1)*out['output_detections'].shift(2)) - out['output_detections'] * out['input_blinks'].shift(-1) * out['output_detections'].shift(-1)

    return out


correcting_funcs = {
    'ID': correct_ID,
    'FN': correct_FN,
    'FP': correct_FP,
}

def get_scores(timeline):
    return pd.DataFrame({
        ind: {
            'track_length': len(track),
            'TP_100':     (track['input_blinks'] * (1* octagonal_output.eq(4))).sum(),#(1*track['output_detections'].rolling(3).apply(lambda x: all(x == (1,0,0)), raw=True).shift(-1))).sum(),
            'TP_010':     (track['input_blinks'] * (1* octagonal_output.eq(2))).sum(),#(1*track['output_detections'].rolling(3).apply(lambda x: all(x == (0,1,0)), raw=True).shift(-1))).sum(),
            'TP_001':     (track['input_blinks'] * (1* octagonal_output.eq(1))).sum(),#(1*track['output_detections'].rolling(3).apply(lambda x: all(x == (0,0,1)), raw=True).shift(-1))).sum(),
            'TP_110':     (track['input_blinks'] * (1* octagonal_output.eq(6))).sum(),#(1*track['output_detections'].rolling(3).apply(lambda x: all(x == (1,1,0)), raw=True).shift(-1))).sum(),
            'TP_101':     (track['input_blinks'] * (1* octagonal_output.eq(5))).sum(),#(1*track['output_detections'].rolling(3).apply(lambda x: all(x == (1,0,1)), raw=True).shift(-1))).sum(),
            'TP_011':     (track['input_blinks'] * (1* octagonal_output.eq(3))).sum(),#(1*track['output_detections'].rolling(3).apply(lambda x: all(x == (0,1,1)), raw=True).shift(-1))).sum(),
            'TP_111':     (track['input_blinks'] * (1* octagonal_output.eq(7))).sum(),#(1*track['output_detections'].rolling(3).apply(lambda x: all(x == (1,1,1)), raw=True).shift(-1))).sum(),
            'FN_000':     (track['input_blinks'] * (1* octagonal_output.eq(0))).sum(),#(1*track['output_detections'].rolling(3).apply(lambda x: all(x == (1,1,1)), raw=True).shift(-1))).sum(),
            'TN_100': ((1-track['input_blinks']) * (1* octagonal_output.eq(4))).sum(),#(1*track['output_detections'].rolling(3).apply(lambda x: all(x == (1,0,0)), raw=True).shift(-1))).sum(),
            'FP_010': ((1-track['input_blinks']) * (1* octagonal_output.eq(2))).sum(),#(1*track['output_detections'].rolling(3).apply(lambda x: all(x == (0,1,0)), raw=True).shift(-1))).sum(),
            'TN_001': ((1-track['input_blinks']) * (1* octagonal_output.eq(1))).sum(),#(1*track['output_detections'].rolling(3).apply(lambda x: all(x == (0,0,1)), raw=True).shift(-1))).sum(),
            'FP_110': ((1-track['input_blinks']) * (1* octagonal_output.eq(6))).sum(),#(1*track['output_detections'].rolling(3).apply(lambda x: all(x == (1,1,0)), raw=True).shift(-1))).sum(),
            'TN_101': ((1-track['input_blinks']) * (1* octagonal_output.eq(5))).sum(),#(1*track['output_detections'].rolling(3).apply(lambda x: all(x == (1,0,1)), raw=True).shift(-1))).sum(),
            'FP_011': ((1-track['input_blinks']) * (1* octagonal_output.eq(3))).sum(),#(1*track['output_detections'].rolling(3).apply(lambda x: all(x == (0,1,1)), raw=True).shift(-1))).sum(),
            'FP_111': ((1-track['input_blinks']) * (1* octagonal_output.eq(7))).sum(),#(1*track['output_detections'].rolling(3).apply(lambda x: all(x == (1,1,1)), raw=True).shift(-1))).sum(),
            'TN_000': ((1-track['input_blinks']) * (1* octagonal_output.eq(0))).sum(),#(1*track['output_detections'].rolling(3).apply(lambda x: all(x == (1,1,1)), raw=True).shift(-1))).sum(),
            'input_pulses': track['input_blinks'].sum(),
            'output_pulses': track['output_detections'].sum()- (1*(track['output_detections'].rolling(2).sum() ==2)).sum(),
        }
            for ind, track in ProgressBar()(timeline.groupby(['classifier', 'iteration', 'track_id']))
                for octagonal_output in [track['output_detections'].rolling(3, min_periods=3, center=True).apply(lambda x: 4*x[0] + 2*x[1] + x[2], raw=True).fillna(0).astype(np.int)]
    }).rename_axis(columns=['classifier', 'iteration', 'track_id']).T.assign(
        TP = lambda x: x[[key for key in x.columns if 'TP_' in key]].sum(axis=1),
        FN = lambda x: x['FN_000'], 
        FP = lambda x: x['output_pulses'] - x['TP'],
        TN = lambda x: (x['track_length']-x['output_pulses']) - x['FN'],
        )

class Step(AbstractStep):   

    step_name = 'ITfm'

    required_parameters = ['theoretical_parameters', 'loss_source_determination_method']
    input_files = ['binary_timeline']
    output_files = {'information_overall_empirical': '.pkl.gz', 'information_overall_theoretical': '.pkl.gz', 'information_per_track_empirical': '.pkl.gz', 'information_per_track_theoretical': '.pkl.gz'}


    def perform(self, **kwargs):
        theoretical_min = kwargs['theoretical_parameters']['min']
        theoretical_exp_mean = kwargs['theoretical_parameters']['exp_mean']
        minutes_per_timepoint = kwargs['theoretical_parameters']['minutes_per_timepoint']
        method = kwargs['loss_source_determination_method']


        print('------------ COMPUTING INFORMATION TRANSMISSION RATE (FULL 3bit CM)------------')
        isSequential = method[:21] == 'sequential_correction' or method == 'sequential_averaged'
        assert method in ['full_matrix', 'fixing_ID_FN_FP'] or isSequential



        if method == 'full_matrix':
            def loss_due_to_inaccuracy(x):
                keys = [
                        'TP_100', 'TP_010', 'TP_001', 'TP_110', 'TP_101', 'TP_011', 'TP_111', 'FN_000', 
                        'TN_100', 'FP_010', 'TN_001', 'FP_110', 'TN_101', 'FP_011', 'FP_111', 'TN_000',
                    ]
                return (
                    sum(plogp(x[key]) for key in keys) - sum(plogp(sum(x[key] for key in keys if pattern in key)) for a in (0,1) for b in (0,1) for c in (0,1) for pattern in [f"{a}{b}{c}"])
                    - loss_due_to_FP(x) - loss_due_to_FN(x))
                

            def loss_due_to_FP(x):
                return  plogp(x['TP']) +plogp(x['FP'])- plogp(x['TP'] + x['FP']) # + (plogp(x['TP'] + x['FN']) - plogp(x['TP'] + x['FN'] + x['FP']) - plogp(x['TP']) + plogp(x['TP'] + x['FP']))

            def loss_due_to_FN(x):
                return plogp(x['FN'])+plogp(x['TN'])- plogp(x['FN']+x['TN']) #- (plogp(x['TP'] + x['FN']) - plogp(x['TP'] + x['FN'] + x['FP']) - plogp(x['TP']) + plogp(x['TP'] + x['FP']))
                
            def conditional_entropy(x):
                return loss_due_to_inaccuracy(x) + loss_due_to_FP(x) + loss_due_to_FN(x)


        elif isSequential:
            if method != 'sequential_averaged':
                error_correction_sequence = method[22:].split('_')
                assert len(error_correction_sequence) == 3 and all(kind_of_error in error_correction_sequence for kind_of_error in ('ID', 'FN', 'FP'))
                print(error_correction_sequence)
            def conditional_entropy (x):
                return (
                    +bernoulli_entropy_rate(x['TP_100'], x['TN_100'])
                    +bernoulli_entropy_rate(x['TP_010'], x['FP_010'])
                    +bernoulli_entropy_rate(x['TP_001'], x['TN_001'])
                    +bernoulli_entropy_rate(x['TP_110'], x['FP_110'])
                    +bernoulli_entropy_rate(x['TP_011'], x['FP_011'])
                    +bernoulli_entropy_rate(x['TP_101'], x['TN_101'])
                    +bernoulli_entropy_rate(x['TP_111'], x['FP_111'])
                    +bernoulli_entropy_rate(x['FN_000'], x['TN_000'])
                )


        
        elif method == 'fixing_ID_FN_FP':

            # def correct_ID(x: pd.DataFrame):
            #     out = x.copy()
            #     out['TP_010'] = x['TP_010'] + x['TP_100'] + x['TP_001']
            #     out['FP_010'] = x['FP_010'] - x['TP_100'] - x['TP_001']
            #     out['TP_100'] = 0
            #     out['TN_100'] = x['TN_100'] + x['TP_100']
            #     out['TP_001'] = 0
            #     out['TN_001'] = x['TN_001'] + x['TP_001']

            #     out['TP_011'] = x['TP_011'] + x['TP_101']
            #     out['FP_110'] = x['FP_110'] + x['TP_101']
            #     out['TN_000'] = x['TN_000'] + x['TP_101']
            #     out['FP_010'] = out['FP_010'] - 2*x['TP_101']
            #     out['TP_101'] = 0
                
            #     return out

            # def correct_FN(x: pd.DataFrame):
            #     out = x.copy()
            #     out['TP_010'] = x['TP_010'] + x['FN_000']
            #     out['TN_100'] = x['TN_100'] + x['FN_000']
            #     out['TN_001'] = x['TN_001'] + x['FN_000']
            #     out['FN_000'] = 0
            #     out['TP_000'] = x['TP_000'] - 2*x['FN_000']
            #     return out

            # def correct_FP_010(x: pd.DataFrame):
            #     out = x.copy()
            #     out['FP_010'] = x['TP_100'] + x['TP_001']
            #     out['TN_000'] = x['TN_000'] + 3*(x['FP_010'] - x['TP_100'] + x['TP_001'])
            #     out['TN_100'] = x['TN_100'] - (x['FP_010'] - x['TP_100'] + x['TP_001'])
            #     out['TN_001'] = x['TN_001'] - (x['FP_010'] - x['TP_100'] + x['TP_001'])

            #     return out

            # def correct_TP_110(x: pd.DataFrame):
            #     out = x.copy()
            #     out['TP_110'] = 0
            #     out['TP_010'] = x['TP_010'] + x['TP_110']
            #     out['FP_011'] = x['FP_011'] - x['TP_110']
            #     out['TN_000'] = x['TN_000'] + x['TP_110']
            #     return out

            # def correct_TP_011(x: pd.DataFrame):
            #     out = x.copy()
            #     out['TP_011'] = 0
            #     out['TP_010'] = x['TP_010'] + x['TP_011']
            #     out['FP_110'] = x['FP_110'] - x['TP_011']
            #     out['TN_000'] = x['TN_000'] + x['TP_011']
            #     return out

            # def correct_FP_011_110(x: pd.DataFrame):
            #     out = x.copy()
            #     out['FP_011'] = x['TP_110'] + x['TP_111']
            #     out['TP_010'] = x['TP_'] + x['TP_011']
            #     out['FP_110'] = x['FP_110'] - x['TP_011']
            #     out['TN_000'] = x['TN_000'] + 4*(x['FP_011'] - x['TP_110'] - x['TP_111'])
            #     return out





            # def correct_FP(x: pd.DataFrame):
            #     out = x.copy()
            #     out['TP_010'] = x['TP_010'] + x['TP_011'] + x['TP_110'] + x['TP_111']
            #     out['FP_010'] = x['TP_100'] + x['TP_001']
            #     out['TP_011'] = 0
            #     out['TP_110'] = 0
            #     out['TP_111'] = 0
            #     out['TP_101'] = 0
            #     out['TP_100'] = x['TP_100'] + x['TP_101'] 
            #     out['TN_100'] = x['TN_100'] - (x['FP_010'] - x['TP_100'] - x['TP_001']) - (x['FP_110'] - x['TP_011'] - x['TP_111']- x['FP_111']) - x['FP_111']
            #     out['TN_001'] = x['TN_001'] - (x['FP_010'] - x['TP_100'] - x['TP_001']) - (x['FP_011'] - x['TP_110'] - x['TP_111']- x['FP_111']) - x['FP_111']
            #     out['TN_000'] = x['TN_000'] + 3*(x['FP_010'] - x['TP_100'] - x['TP_001']) + x['TP_011'] + 2*x['TP_101'] + x['TP_110'] + 2* (x['FP_110'] - x['TP_011'] - x['TP_111'] - x['FP_111'] ) + 2* (x['FP_011'] - x['TP_110'] - x['TP_111']- x['FP_111']) + 2*x['TP_111'] + 5*x['FP_111']

            #     return correct_FP_010(correct_TP_110(correct_011(correct_101(x))))

            def conditional_entropy (x):
                return (
                    +bernoulli_entropy_rate(x['TP_100'], x['TN_100'])
                    +bernoulli_entropy_rate(x['TP_010'], x['FP_010'])
                    +bernoulli_entropy_rate(x['TP_001'], x['TN_001'])
                    +bernoulli_entropy_rate(x['TP_110'], x['FP_110'])
                    +bernoulli_entropy_rate(x['TP_011'], x['FP_011'])
                    +bernoulli_entropy_rate(x['TP_101'], x['TN_101'])
                    +bernoulli_entropy_rate(x['TP_111'], x['FP_111'])
                    +bernoulli_entropy_rate(x['FN_000'], x['TN_000'])
                )

            def loss_due_to_inaccuracy(x):
                return (
                    +bernoulli_entropy_rate(x['TP_100'], x['TN_100'])
                    +bernoulli_entropy_rate(x['TP_010'], x['FP_010'])
                    +bernoulli_entropy_rate(x['TP_001'], x['TN_001'])
                    # -bernoulli_entropy_rate(x['TP_010'] + x['TP_100'] + x['TP_001'],   x['FP_010'] - x['TP_100'] - x['TP_001'])
                    # -bernoulli_entropy_rate(0, x['TN_100'] + x['TP_100'])
                    # -bernoulli_entropy_rate(0, x['TN_001'] + x['TP_001'])
                    
                    +bernoulli_entropy_rate(x['TP_101'], x['TN_101'])
                    +bernoulli_entropy_rate(x['TP_011'], x['FP_011'])
                    +bernoulli_entropy_rate(x['TP_110'], x['FP_110'])
                    +bernoulli_entropy_rate(x['FN_000'], x['TN_000'])
                    # +bernoulli_entropy_rate(x['TP_010'] + x['TP_100'] + x['TP_001'],   x['FP_010'] - x['TP_100'] - x['TP_001'])
                    -bernoulli_entropy_rate(x['TP_011'] + x['TP_101'],   x['FP_011'])
                    -bernoulli_entropy_rate(x['TP_110'],   x['FP_110'] + x['TP_101'])
                    -bernoulli_entropy_rate(x['FN_000'],   x['TN_000'] + x['TP_101'])
                    -bernoulli_entropy_rate(x['TP_010'] + x['TP_100'] + x['TP_001'],   x['FP_010'] - x['TP_100'] - x['TP_001'] - 2*x['TP_101'])
                    # -bernoulli_entropy_rate(0, x['TN_101'])
                )

            def loss_due_to_FN(x):
                return (
                    +bernoulli_entropy_rate(x['TP_010'] + x['TP_100'] + x['TP_001'],   x['FP_010'] - x['TP_100'] - x['TP_001'] - 2*x['TP_101'])
                    +bernoulli_entropy_rate(x['FN_000'], x['TN_000'])
                    -bernoulli_entropy_rate(x['TP_010'] + x['TP_100'] + x['TP_001'] + x['FN_000'],   x['FP_010'] - x['TP_100'] - x['TP_001'] - 2*x['TP_101'])
                    # -bernoulli_entropy_rate(0, x['TN_000'])

                    
                ) 
            def loss_due_to_FP(x):
                return (
                    +bernoulli_entropy_rate(x['TP_010'] + x['TP_100'] + x['TP_001'] + x['FN_000'],   x['FP_010'] - x['TP_100'] - x['TP_001'] - 2*x['TP_101'])
                    +bernoulli_entropy_rate(x['TP_011'] + x['TP_101'],   x['FP_011'])
                    +bernoulli_entropy_rate(x['TP_110'],   x['FP_110'] + x['TP_101'])
                    +bernoulli_entropy_rate(x['TP_111'],   x['FP_111'])
                    # +bernoulli_entropy_rate(0, x['TN_000'])
                    # +bernoulli_entropy_rate(0, x['TN_001'] + x['TP_001'])
                    # +bernoulli_entropy_rate(0, x['TN_100'] + x['TP_100'])

                    # -bernoulli_entropy_rate(x['TP_010'] + x['TP_100'] + x['TP_001'] + x['FN_000'] + (x['TP_011'] + x['TP_101']) + x['TP_110'] + x['TP_111'],   0)
                    # -bernoulli_entropy_rate(0,   0)
                    # -bernoulli_entropy_rate(0,   0)
                    # -bernoulli_entropy_rate(0, x['TN_000'] + (x['FP_010'] - x['TP_100'] - x['TP_001'] - 2*x['TP_101']) + (x['TP_011'] + x['TP_101']) + x['TP_110'] + 2* (x['FP_110'] - x['TP_011'] - x['TP_111'] - x['FP_111'] ) + 2* (x['FP_011'] - x['TP_110'] - x['TP_111']- x['FP_111']) + 2*x['TP_111'] + 5*x['FP_111'])
                    # -bernoulli_entropy_rate(0, x['TN_001'] + x['TP_001'] - (x['FP_110'] - x['TP_011'] - x['TP_111']- x['FP_111']) - x['FP_111'])
                    # -bernoulli_entropy_rate(0, x['TN_100'] + x['TP_100'] - (x['FP_011'] - x['TP_110'] - x['TP_111']- x['FP_111']) - x['FP_111'])
                    

                )


        binary_timeline : pd.DataFrame = self.load_file('binary_timeline')

        theoretical_mean = theoretical_min + theoretical_exp_mean
        
        theoretical_input_frequency : float= 1/theoretical_mean# if not regular else 0.5
        theoretical_input_entropy : float = input_entropy(theoretical_min, theoretical_exp_mean)# if not regular else 1
        theoretical_input_entropy_assuming_poisson : float = input_entropy(0, theoretical_mean)# if not regular else 1

        def norm_for_theoretical_input(x):
            dictionary = {
                    **{key: x[key] * theoretical_input_frequency / x['input_pulses'] for key in [
                        'TP', 'FN', 'TP_100', 'TP_010', 'TP_001', 'TP_110', 'TP_101', 'TP_011', 'TP_111', 'FN_000', 'input_pulses']},
                     **{key: x[key] * (1-theoretical_input_frequency) / (1-x['input_pulses']) for key in [
                        'FP', 'TN', 'TN_100', 'FP_010', 'TN_001', 'FP_110', 'TN_101', 'FP_011', 'FP_111', 'TN_000']},
                    
                    # 'input_pulses': x['input_pulses'] * theoretical_input_frequency / x['input_pulses'],
                    'output_pulses': (x['TP']) * theoretical_input_frequency / x['input_pulses'] + x['FP'] * (1-theoretical_input_frequency) / (1-x['input_pulses']),
                }
            if type(x) == pd.DataFrame: return pd.DataFrame(dictionary, index = x.index)
            else: return pd.Series(dictionary)
        

        print("Scoring", end='... ', flush=True)
        
         
        def get_per_minute_empirical(scores) -> pd.DataFrame:
            return scores.div(scores['track_length'], axis=0).assign(
                input_entropy=lambda x: input_entropy(theoretical_min, 1/x['input_pulses'] - theoretical_min),
                conditional_entropy=conditional_entropy,
            )


        def add_fields_theoretical(scores) -> pd.DataFrame:
            return norm_for_theoretical_input(scores).assign(
                input_entropy=lambda x: theoretical_input_entropy,
                conditional_entropy=conditional_entropy,
            )

        def sum_and_normalize(x) -> pd.Series:
            return x.sum()/x['track_length'].sum()


        
        if isSequential and method != 'sequential_averaged':
            fixed_timelines = [binary_timeline]
            for kind_of_error in error_correction_sequence:
                print(f"Correcting for {kind_of_error}...", flush=True)
                fixed_timelines.append(correcting_funcs[kind_of_error](fixed_timelines[-1]))

            scores_fixed = [get_scores(timeline).pipe(lambda x: x[(x['input_pulses'] > 0) & (x['input_pulses'] < x['track_length'] )]) for timeline in fixed_timelines]
            scores_fixed_empirical = [sum_and_normalize(sc) for sc in scores_fixed]
            scores_fixed_theoretical = [sum_and_normalize(sc).pipe(norm_for_theoretical_input) for sc in scores_fixed]
            scores_fixed_empirical_per_track = [get_per_minute_empirical(sc).groupby('track_id').mean() for sc in scores_fixed]
            scores_fixed_theoretical_per_track = [get_per_minute_empirical(sc).pipe(add_fields_theoretical).groupby('track_id').mean() for sc in scores_fixed]
            def losses(sc):
                return {
                    kind_of_error:  conditional_entropy(scores_fixed_prev) - conditional_entropy(scores_fixed_next)
                        for kind_of_error, scores_fixed_next, scores_fixed_prev in zip(error_correction_sequence, sc[1:], sc[:-1])
                    }
            losses_empirical = losses(scores_fixed_empirical)
            losses_theoretical = losses(scores_fixed_theoretical)
            losses_empirical_per_track = losses(scores_fixed_empirical_per_track)
            losses_theoretical_per_track = losses(scores_fixed_theoretical_per_track)

        if method == 'sequential_averaged':
            error_correction_sequence = list(np.sort(['ID', 'FP', 'FN']))
            fixed_timelines = {tuple(): binary_timeline}
            for k in range(3):
                for subset in combinations(error_correction_sequence, k+1):
                    timeline = correcting_funcs[subset[-1]](fixed_timelines[subset[:-1]])
                    fixed_timelines = {**fixed_timelines, subset: timeline}
            scores_fixed = {key: get_scores(sc) for key, sc in fixed_timelines.items()}
            scores_fixed_empirical = {key: sum_and_normalize(sc) for key,sc in scores_fixed.items()}
            scores_fixed_theoretical = {key: sum_and_normalize(sc).pipe(norm_for_theoretical_input) for key,sc in scores_fixed.items()}
            scores_fixed_empirical_per_track = {key: get_per_minute_empirical(sc).groupby('track_id').mean() for key,sc in scores_fixed.items()}
            scores_fixed_theoretical_per_track = {key: get_per_minute_empirical(sc).pipe(add_fields_theoretical).groupby('track_id').mean() for key,sc in scores_fixed.items()}
            
            def losses(sc):
                return {
                    kind_of_error:  (lambda x: sum(x)/len(x))([
                        conditional_entropy(scores_fixed_prev) - conditional_entropy(scores_fixed_next) 
                            for perm in permutations(error_correction_sequence) 
                            for i in (perm.index(kind_of_error),)  
                            for scores_fixed_next in (sc[tuple(np.sort(perm[:(i+1)]))], )
                            for scores_fixed_prev in (sc[tuple(np.sort(perm[:(i)]))], )
                        ])
                        for kind_of_error in error_correction_sequence
                    }
            losses_empirical = losses(scores_fixed_empirical)
            losses_theoretical = losses(scores_fixed_theoretical)
            losses_empirical_per_track = losses(scores_fixed_empirical_per_track)
            losses_theoretical_per_track = losses(scores_fixed_theoretical_per_track)
            

        scores = get_scores(binary_timeline) if not isSequential else scores_fixed[tuple()]
        scores = scores[(scores['input_pulses'] > 0) & (scores['input_pulses'] < scores['track_length'] )]

        print(scores.describe().T)
        print("Normalizing", end='... ', flush=True)
        scores_per_minute = get_per_minute_empirical(scores)
        theoretical_scores_per_minute = add_fields_theoretical(scores_per_minute)
        print("Done", flush=True)



        overall_empirical : pd.Series = scores.pipe(sum_and_normalize)
        overall_empirical['input_entropy'] = input_entropy(theoretical_min, 1/overall_empirical['input_pulses'] - theoretical_min)
        overall_empirical['conditional_entropy'] = conditional_entropy(overall_empirical)
        overall_empirical['information lost due to false detections[b/timepoint]'] = loss_due_to_FP(overall_empirical)             if not isSequential else losses_empirical['FP']
        overall_empirical['information lost due to missed pulses[b/timepoint]'] = loss_due_to_FN(overall_empirical)                if not isSequential else losses_empirical['FN']
        overall_empirical['information lost due to inacurate detections[b/timepoint]'] = loss_due_to_inaccuracy(overall_empirical) if not isSequential else losses_empirical['ID']
        overall_empirical['channel_capacity[b/timepoint]'] = overall_empirical['input_entropy'] - overall_empirical['conditional_entropy']
        overall_empirical['channel_capacity_assuming_poisson[b/timepoint]'] = input_entropy(0, 1/overall_empirical['input_pulses']) - overall_empirical['conditional_entropy']
        overall_empirical['channel_capacity[b/h]'] = 60/minutes_per_timepoint*overall_empirical['channel_capacity[b/timepoint]']
        overall_empirical['channel_capacity_assuming_poisson[b/h]'] = 60/minutes_per_timepoint*overall_empirical['channel_capacity_assuming_poisson[b/timepoint]']
        overall_empirical['channel_capacity[min/b]'] = minutes_per_timepoint/overall_empirical['channel_capacity[b/timepoint]']
        overall_empirical['channel_capacity_assuming_poisson[min/b]'] = minutes_per_timepoint/overall_empirical['channel_capacity_assuming_poisson[b/timepoint]']
        # overall_empirical['1st'], overall_empirical['2nd'], overall_empirical['3rd'], overall_empirical['4th'] = (lambda x: (
        #     sum(plogp(x[key]) for key in keys) , 
        #     sum(plogp(sum(x[key] for key in keys if pattern in key)) for a in (0,1) for b in (0,1) for c in (0,1) for pattern in [f"{a}{b}{c}"]), 
        #      plogp(x['TP'] + x['FP']) + plogp(x['FN']+x['TN']),
        #       sum(plogp(x[key]) for key in ['TP', 'FN', 'FP', 'TN'])
        #       ))(overall_empirical)

        # print((lambda x: pd.DataFrame([(x[keys[i]], x[keys[i+8]], x[keys[i]]/ (x[keys[i]]+x[keys[i+8]]), plogp(x[keys[i]]) + plogp(x[keys[i+8]]) - plogp(x[keys[i]] +x[keys[i+8]])) for i in range(8)]).pipe(lambda df: df.append(df.sum(), ignore_index=True)))(overall_empirical))
        

        overall_theoretical : pd.Series  = norm_for_theoretical_input(overall_empirical)
        overall_theoretical['input_entropy'] = theoretical_input_entropy
        overall_theoretical['conditional_entropy'] = conditional_entropy(overall_theoretical)
        overall_theoretical['information lost due to missed pulses[b/timepoint]'] = loss_due_to_FN(overall_theoretical)                 if not isSequential else losses_theoretical['FN']
        overall_theoretical['information lost due to false detections[b/timepoint]'] = loss_due_to_FP(overall_theoretical)              if not isSequential else losses_theoretical['FP']
        overall_theoretical['information lost due to inacurate detections[b/timepoint]'] = loss_due_to_inaccuracy(overall_theoretical)  if not isSequential else losses_theoretical['ID']
        overall_theoretical['channel_capacity[b/timepoint]'] = overall_theoretical['input_entropy'] - overall_theoretical['conditional_entropy']
        overall_theoretical['channel_capacity_assuming_poisson[b/timepoint]'] = theoretical_input_entropy_assuming_poisson - overall_theoretical['conditional_entropy']
        overall_theoretical['channel_capacity[b/h]'] = 60/minutes_per_timepoint*overall_theoretical['channel_capacity[b/timepoint]']
        overall_theoretical['channel_capacity_assuming_poisson[b/h]'] = 60/minutes_per_timepoint*overall_theoretical['channel_capacity_assuming_poisson[b/timepoint]']
        overall_theoretical['channel_capacity[min/b]'] = minutes_per_timepoint/overall_theoretical['channel_capacity[b/timepoint]']
        overall_theoretical['channel_capacity_assuming_poisson[min/b]'] = minutes_per_timepoint/overall_theoretical['channel_capacity_assuming_poisson[b/timepoint]']
        
        per_track_empirical = scores_per_minute.groupby('track_id').mean()
        per_track_empirical['information lost due to false detections[b/timepoint]'] = loss_due_to_FP(per_track_empirical)              if not isSequential else losses_empirical_per_track['FP']
        per_track_empirical['information lost due to missed pulses[b/timepoint]'] = loss_due_to_FN(per_track_empirical)                 if not isSequential else losses_empirical_per_track['FN']
        per_track_empirical['information lost due to inacurate detections[b/timepoint]'] = loss_due_to_inaccuracy(per_track_empirical)  if not isSequential else losses_empirical_per_track['ID']
        per_track_empirical['channel_capacity[b/timepoint]'] = input_entropy(theoretical_min, 1/per_track_empirical['input_pulses'] - theoretical_min)- per_track_empirical['conditional_entropy']
        per_track_empirical['channel_capacity_assuming_poisson[b/timepoint]'] = input_entropy(0, 1/per_track_empirical['input_pulses']) - per_track_empirical['conditional_entropy']
        per_track_empirical['channel_capacity[b/h]'] = 60/minutes_per_timepoint*per_track_empirical['channel_capacity[b/timepoint]']
        per_track_empirical['channel_capacity_assuming_poisson[b/h]'] = 60/minutes_per_timepoint*per_track_empirical['channel_capacity_assuming_poisson[b/timepoint]']
        per_track_empirical['channel_capacity[min/b]'] = minutes_per_timepoint/per_track_empirical['channel_capacity[b/timepoint]']
        per_track_empirical['channel_capacity_assuming_poisson[min/b]'] = minutes_per_timepoint/per_track_empirical['channel_capacity_assuming_poisson[b/timepoint]']
        
        per_track_theoretical = theoretical_scores_per_minute.groupby('track_id').mean()
        per_track_theoretical['information lost due to false detections[b/timepoint]'] = loss_due_to_FP(per_track_theoretical)              if not isSequential else losses_theoretical_per_track['FP']
        per_track_theoretical['information lost due to missed pulses[b/timepoint]'] = loss_due_to_FN(per_track_theoretical)                 if not isSequential else losses_theoretical_per_track['FN']
        per_track_theoretical['information lost due to inacurate detections[b/timepoint]'] = loss_due_to_inaccuracy(per_track_theoretical)  if not isSequential else losses_theoretical_per_track['ID']
        per_track_theoretical['channel_capacity[b/timepoint]'] = theoretical_input_entropy - per_track_theoretical['conditional_entropy']
        per_track_theoretical['channel_capacity_assuming_poisson[b/timepoint]'] = theoretical_input_entropy_assuming_poisson - per_track_theoretical['conditional_entropy']
        per_track_theoretical['channel_capacity[b/h]'] = 60/minutes_per_timepoint*per_track_theoretical['channel_capacity[b/timepoint]']
        per_track_theoretical['channel_capacity_assuming_poisson[b/h]'] = 60/minutes_per_timepoint*per_track_theoretical['channel_capacity_assuming_poisson[b/timepoint]']
        per_track_theoretical['channel_capacity[min/b]'] = minutes_per_timepoint/per_track_theoretical['channel_capacity[b/timepoint]']
        per_track_theoretical['channel_capacity_assuming_poisson[min/b]'] = minutes_per_timepoint/per_track_theoretical['channel_capacity_assuming_poisson[b/timepoint]']
        
        print(overall_empirical)


        self.save_file(overall_empirical, 'information_overall_empirical')
        self.save_file(overall_theoretical, 'information_overall_theoretical')
        self.save_file(per_track_empirical, 'information_per_track_empirical')
        self.save_file(per_track_theoretical, 'information_per_track_theoretical')






    
