import pandas as pd
from itertools import permutations

from core.step_manager import AbstractStep
from utils.math_utils import plogp




def factorial(n: int) -> int:
    f = 1
    for i in range(1,n+1):
        f *= i
    return f

def compute_mi(conf_m: pd.Series) -> int:
    return (plogp(conf_m.groupby('y_true').sum()).sum() + plogp(conf_m.groupby('y_pred').sum()).sum() - plogp(conf_m.groupby(['y_true', 'y_pred']).sum()).sum() - plogp(conf_m.sum())) / conf_m.sum()
 

def correct_ID(conf_m: pd.Series):
    new_matrix = conf_m.copy()

    new_matrix[0, 2] -= conf_m[1, 1] + conf_m[1, 4] + 2 * conf_m[1, 5]
    new_matrix[1, 4] = 0
    new_matrix[1, 1] = 0
    new_matrix[1, 5] = 0

    new_matrix[1, 2] += conf_m[1, 1] + conf_m[1, 4] + conf_m[1, 5]
    new_matrix[0, 4] += conf_m[1, 4]
    new_matrix[0, 1] += conf_m[1, 1]
    new_matrix[0, 0] += 2 * conf_m[1, 5]

    return new_matrix.round(2)


def correct_FN(conf_m: pd.Series):
    new_matrix = conf_m.copy()

    true_positive = (conf_m[1, 1] + conf_m[1, 2] + conf_m[1, 4] + conf_m[1, 5])

    early_detection_ratio = conf_m[1, 4] / true_positive
    timely_detection_ratio = conf_m[1, 2] / true_positive
    late_detection_ratio = conf_m[1, 1] / true_positive
    early_and_late_detection_ratio = conf_m[1, 5] / true_positive

    false_negatives = conf_m[1, 0]

    new_matrix[1, 0] -= false_negatives

    new_matrix[0, 4] += false_negatives * timely_detection_ratio
    new_matrix[1, 2] += false_negatives * timely_detection_ratio
    new_matrix[0, 1] += false_negatives * timely_detection_ratio
    
    new_matrix[1, 4] += false_negatives * early_detection_ratio
    new_matrix[0, 2] += false_negatives * early_detection_ratio
    new_matrix[0, 1] += false_negatives * early_detection_ratio

    new_matrix[0, 4] += false_negatives * late_detection_ratio
    new_matrix[0, 2] += false_negatives * late_detection_ratio
    new_matrix[1, 1] += false_negatives * late_detection_ratio

    new_matrix[0, 4] += false_negatives * early_and_late_detection_ratio
    new_matrix[1, 5] += false_negatives * early_and_late_detection_ratio
    new_matrix[0, 2] += 2 * false_negatives * early_and_late_detection_ratio
    new_matrix[0, 1] += false_negatives * early_and_late_detection_ratio
    
    new_matrix[0, 0] -= 2 * false_negatives * (1 + early_and_late_detection_ratio)

    return new_matrix


def correct_FP(conf_m: pd.Series):
    new_matrix = conf_m.copy()

    false_positives = conf_m[0, 2] - conf_m[1, 4] - conf_m[1, 1] - 2 * conf_m[1, 5]

    new_matrix[0, 4] -= false_positives
    new_matrix[0, 2] -= false_positives
    new_matrix[0, 1] -= false_positives

    new_matrix[0, 0] += 3 * false_positives

    return new_matrix


def correct_FN_r1(conf_m: pd.Series):
    new_matrix = conf_m.copy()
    
    new_matrix[1, 1] += conf_m[1, 0]
    new_matrix[1, 0] = 0

    return new_matrix


def correct_FP_r1(conf_m: pd.Series):
    new_matrix = conf_m.copy()
    
    new_matrix[0, 0] += conf_m[0, 1]
    new_matrix[0, 1] = 0

    return new_matrix

class Step(AbstractStep):

    step_name = 'LS'

    required_parameters = ['r_slice_length', 's_slice_length']
    input_files = ['confusion_matrix']
    output_files = {'loss_sources': '.pkl.gz'}



    def perform(self, **kwargs):
        print('------ESTIMATING LOSS SOURCES ------')

        s_slice_length = kwargs['s_slice_length']
        r_slice_length = kwargs['r_slice_length']

        confusion_matrix: pd.Series = self.load_file('confusion_matrix')


        assert s_slice_length == 1
        assert r_slice_length in (1,3)

        if r_slice_length == 1:
            all_options = [(i, j) for i in [0, 1] for j in [0, 1]]
            corrections = {
                'FN': correct_FN_r1,
                'FP': correct_FP_r1,
            }
        elif r_slice_length == 3:
            all_options = [(i, j) for i in [0, 1] for j in [0,1,2,4,5]]
            corrections = {
                'ID': correct_ID,
                'FN': correct_FN,
                'FP': correct_FP,
            }
        else:
            raise ValueError(f'r_slice_length must be either 1 or 3, not {r_slice_length}')


        confusion_matrix = confusion_matrix.groupby(['y_true', 'y_pred']).sum().reindex(all_options).fillna(0)


        lost_on = {key: 0 for key in corrections.keys()}

        for correction_sequence in permutations(corrections.items()):
            old_conf_m = confusion_matrix.copy()
            for error_type, correction_func in correction_sequence:
                print(old_conf_m)
                print(error_type)
                new_conf_m = correction_func(old_conf_m)
                lost_on[error_type] += compute_mi(new_conf_m) - compute_mi(old_conf_m)
                old_conf_m = new_conf_m
       
        lost_on_df = pd.Series(lost_on) / factorial(len(corrections))

        self.save_file(lost_on_df, 'loss_sources')
