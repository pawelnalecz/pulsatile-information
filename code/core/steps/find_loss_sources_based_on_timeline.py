import pandas as pd
from itertools import permutations

from core.step_manager import AbstractStep
from utils.math_utils import plogp

from utils.math_utils import rolling_binary_sequence_to_int
from utils.data_utils import index_without

from typing import List


def factorial(n: int) -> int:
    f = 1
    for i in range(1,n+1):
        f *= i
    return f


def get_conf_matrix(binary_predictions: pd.DataFrame, s_slice_length: int, r_slice_length: int, all_options: List[int]):
    sliced_predictions = pd.concat({
            ind: trajectory.agg({
                'y_true': lambda x: rolling_binary_sequence_to_int(x, s_slice_length),
                'y_pred': lambda x: rolling_binary_sequence_to_int(x, r_slice_length)
                }).dropna().astype(int)# rolling_binary_sequence_to_int(trajectory, r_slice_length).dropna().astype('int').groupby(['y_true', 'y_pred']).size()
            for ind, trajectory in binary_predictions[['y_true', 'y_pred']].dropna().groupby(index_without(binary_predictions, ['time_point']))
        }, names=index_without(binary_predictions, ['time_point']))
    confusion_matrix = sliced_predictions.groupby(['y_true', 'y_pred']).size().reindex(all_options).fillna(0)
    return confusion_matrix


def compute_mi(conf_m: pd.Series) -> int:
    return (plogp(conf_m.groupby('y_true').sum()).sum() + plogp(conf_m.groupby('y_pred').sum()).sum() - plogp(conf_m.groupby(['y_true', 'y_pred']).sum()).sum() - plogp(conf_m.sum())) / conf_m.sum()


def correct_ID(binary_predictions: pd.DataFrame):
    print('Correcting ID... ', end='', flush=True)
    new_binary_predictions = binary_predictions.copy()

    new_binary_predictions['y_pred'] = (binary_predictions['y_pred'] \
        | (
            binary_predictions['y_true'] \
                & (
                      binary_predictions['y_pred'].groupby(index_without(binary_predictions, ['time_point'])).shift(-1, fill_value=0)
                    | binary_predictions['y_pred'].groupby(index_without(binary_predictions, ['time_point'])).shift(1, fill_value=0)))) \
        & (1 - (
                binary_predictions['y_true'].groupby(index_without(binary_predictions, ['time_point'])).shift(-1, fill_value=0) \
              | binary_predictions['y_true'].groupby(index_without(binary_predictions, ['time_point'])).shift(1, fill_value=0))
            )
    
    return new_binary_predictions


def correct_FN_r3(binary_predictions: pd.DataFrame):
    print('Correcting FN... ', end='', flush=True)
    new_binary_predictions = binary_predictions.copy()

    new_binary_predictions['y_pred'] = binary_predictions['y_pred'] \
        | (binary_predictions['y_true']
            & (1 - binary_predictions['y_pred'].groupby(index_without(binary_predictions, ['time_point'])).shift(-1, fill_value=0))
            & (1 - binary_predictions['y_pred'].groupby(index_without(binary_predictions, ['time_point'])).shift(1, fill_value=0)))

    return new_binary_predictions


def correct_FP_r3(binary_predictions: pd.DataFrame):
    print('Correcting FP... ', end='', flush=True)
    new_binary_predictions = binary_predictions.copy()

    new_binary_predictions['y_pred'] = binary_predictions['y_pred'] \
        & (binary_predictions['y_true']
           | binary_predictions['y_true'].groupby(index_without(binary_predictions, ['time_point'])).shift(-1, fill_value=0)
           | binary_predictions['y_true'].groupby(index_without(binary_predictions, ['time_point'])).shift(1, fill_value=0))

    return new_binary_predictions


def correct_FN(binary_predictions: pd.DataFrame):
    print('Correcting FN... ', end='', flush=True)
    new_binary_predictions = binary_predictions.copy()

    new_binary_predictions['y_pred'] = binary_predictions['y_pred'] | binary_predictions['y_true']

    return new_binary_predictions


def correct_FP(binary_predictions: pd.DataFrame):
    print('Correcting FP... ', end='', flush=True)
    new_binary_predictions = binary_predictions.copy()

    new_binary_predictions['y_pred'] = binary_predictions['y_pred'] & binary_predictions['y_true']

    return new_binary_predictions


class Step(AbstractStep):

    step_name = 'LSt'

    required_parameters = ['r_slice_length', 's_slice_length']
    input_files = ['binary_predictions']
    output_files = {'loss_sources': '.pkl.gz'}



    def perform(self, **kwargs):
        print('------ESTIMATING LOSS SOURCES ------')

        s_slice_length = kwargs['s_slice_length']
        r_slice_length = kwargs['r_slice_length']

        binary_predictions: pd.DataFrame = self.load_file('binary_predictions')


        assert s_slice_length == 1
        assert r_slice_length in (1,3)

        if r_slice_length == 1:
            all_options = [(i, j) for i in [0, 1] for j in [0, 1]]
            corrections = {
                'FN': correct_FN,
                'FP': correct_FP,
            }
        elif r_slice_length == 3:
            all_options = [(i, j) for i in [0, 1] for j in [0,1,2,4,5]]
            corrections = {
                'ID': correct_ID,
                'FN': correct_FN_r3,
                'FP': correct_FP_r3,
            }
        else:
            raise ValueError(f'r_slice_length must be either 1 or 3, not {r_slice_length}')



        lost_on = {key: 0 for key in corrections.keys()}

        def compute_mi_from_binary_predictions(binary_pred: pd.DataFrame):
            return compute_mi(get_conf_matrix(binary_pred, s_slice_length=s_slice_length, r_slice_length=r_slice_length, all_options=all_options))

        for correction_sequence in permutations(corrections.items()):
            old_binary_predictions = binary_predictions.copy()
            for error_type, correction_func in correction_sequence:
                new_binary_predictions = correction_func(old_binary_predictions)
                lost_on[error_type] += compute_mi_from_binary_predictions(new_binary_predictions) - compute_mi_from_binary_predictions(old_binary_predictions)
                old_binary_predictions = new_binary_predictions
       
        lost_on_df = pd.Series(lost_on) / factorial(len(corrections))

        self.save_file(lost_on_df, 'loss_sources')



### TESTS

def check(fn, y_true, y_pred, expected_new_y_pred):
    input_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).assign(ind=1, time_point=range(len(y_true))).set_index(['ind', 'time_point'])
    obtained_val = fn(input_df)
    expected_df = pd.DataFrame({'y_true': y_true, 'y_pred': expected_new_y_pred}).assign(ind=1, time_point=range(len(y_true))).set_index(['ind', 'time_point'])
    assert (obtained_val == expected_df).all().all(), f"For input \n{input_df}\n expected \n{expected_df}\n but obtained \n{obtained_val}\n"

for y_true, y_pred, y_pred_new in (
    (
        (0,0,0,1,0,0),
        (0,0,1,0,0,0),
        (0,0,0,1,0,0),
    ),
    (
        (0,0,0,1,0,0),
        (0,0,0,0,1,0),
        (0,0,0,1,0,0),
    ),
    (
        (0,0,0,1,0,0),
        (0,0,0,1,0,0),
        (0,0,0,1,0,0),
    ),
    (
        (1,0,0,1,0,0),
        (0,0,1,0,0,0),
        (0,0,0,1,0,0),
    ),
    (
        (0,0,0,1,0,0,1,0),
        (0,0,1,0,0,0,1,0),
        (0,0,0,1,0,0,1,0),
    ),
    (
        (0,0,0,1,0,0),
        (0,0,1,0,0,1),
        (0,0,0,1,0,1),
    ),
    (
        (0,0,0,1,0,0),
        (0,0,1,0,1,0),
        (0,0,0,1,0,0),
    ),
    (
        (0,0,0,1,0,0),
        (1,0,1,1,0,0),
        (1,0,0,1,0,0),
    ),
): check(correct_ID, y_true, y_pred, y_pred_new)


for y_true, y_pred, y_pred_new in (
    (
        (0,0,0,1,0,1,0,0,1,0,0,1,0,0,1),
        (1,0,1,0,0,0,0,0,1,0,0,0,1,0,0),
        (1,0,1,0,0,1,0,0,1,0,0,0,1,0,1),
    ),
): check(correct_FN_r3, y_true, y_pred, y_pred_new)


for y_true, y_pred, y_pred_new in (
    (
        (0,0,0,1,0,1,0,0,1,0,0,1,0,0,1),
        (1,0,1,0,0,0,0,0,1,0,0,0,1,0,0),
        (0,0,1,0,0,0,0,0,1,0,0,0,1,0,0),
    ),
): check(correct_FP_r3, y_true, y_pred, y_pred_new)


for y_true, y_pred, y_pred_new in (
    (
        (0,0,0,1,0,1,0,0,1,0,0,1,0,0,1),
        (1,0,1,0,0,0,0,0,1,0,0,0,1,0,0),
        (1,0,1,1,0,1,0,0,1,0,0,1,1,0,1),
    ),
): check(correct_FN, y_true, y_pred, y_pred_new)


for y_true, y_pred, y_pred_new in (
    (
        (0,0,0,1,0,1,0,0,1,0,0,1,0,0,1),
        (1,0,1,0,0,0,0,0,1,0,0,0,1,0,0),
        (0,0,0,0,0,0,0,0,1,0,0,0,0,0,0),
    ),
): check(correct_FP, y_true, y_pred, y_pred_new)



