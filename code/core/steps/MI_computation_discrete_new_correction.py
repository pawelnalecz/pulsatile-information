from sklearn.neighbors import NearestNeighbors
import numpy as np

from core.step_manager import AbstractStep
from utils.math_utils import plogp
import pandas as pd
from typing import Literal

from sklearn.model_selection import train_test_split

hinvln2 = 1/np.log(2)/2


def compute_mi(X: np.ndarray, Y: np.ndarray, k: int = 20, correction=True):

    knn_marginal = NearestNeighbors(n_neighbors=k)
    knn_marginal.fit(Y)

    print('Finding neighbors...', end='', flush=True)
    dists, neighs = knn_marginal.kneighbors(Y)
    hinvln2 = 1/np.log(2)/2 if correction else 0

    plogp_with_correction_correction = (lambda x: plogp(x) + hinvln2 *(1 + 1.5 * (x == 1) - 0.75 * (x == 2))) if correction == 'new' else (lambda x: plogp(x) + hinvln2) if correction else plogp

    print('computing conditional entropy...', end='', flush=True)
    conditional_entropy = (
        np.sum([
            np.sum(
                plogp_with_correction_correction(
                    np.unique([X[j] for j in js], return_counts=True)[1]
                )
            ) - hinvln2
            for js in neighs]
        ) / (k * len(X)) + np.log2(k)
    )
    print('computed.', flush=True)
    # print(np.unique(np.array([[u for u in np.unique([X_arr[j] for j in js], return_counts=True)[1]] for js in neighs]), return_counts=True))
    input_entropy = (np.sum(plogp(np.unique(X, return_counts=True)[1]) + hinvln2) - hinvln2 - plogp(len(X))) / len(X)
    print(input_entropy, conditional_entropy, input_entropy - conditional_entropy)
    return input_entropy - conditional_entropy


def split_data(slices: pd.DataFrame, train_on: Literal['same', 'other_tracks', 'other_pulses', 'other_tracks_and_pulses'], test_set_size: int, seed: int):
    print('Splitting train and test data', end='...', flush=True)
    train_test_split_params = {
        'test_size': test_set_size,
        'shuffle': True,
    }
    if train_on == 'same':
        train_index, test_index = train_test_split(slices.index, **train_test_split_params, random_state=seed)
        train_slices = slices.reindex(train_index)
        test_slices = slices.reindex(test_index)
    elif train_on == 'other_pulses':
        train_pulses, test_pulses = train_test_split(slices['pulse_no'].unique(), **train_test_split_params, random_state=seed)
        train_slices = slices[slices['pulse_no'].isin(train_pulses)]
        test_slices = slices[slices['pulse_no'].isin(test_pulses)]
    elif train_on == 'other_tracks':
        train_tracks, test_tracks = train_test_split(slices.index.get_level_values('track_id').unique(), **train_test_split_params, random_state=seed)
        train_slices = slices[slices.index.get_level_values('track_id').isin(train_tracks)]
        test_slices = slices[slices.index.get_level_values('track_id').isin(test_tracks)]
    elif train_on == 'other_tracks_and_pulses':
        train_tracks, test_tracks = train_test_split(slices.index.get_level_values('track_id').unique(), **train_test_split_params, random_state=seed)
        train_pulses, test_pulses = train_test_split(slices['pulse_no'].unique(), **train_test_split_params, random_state=seed)
        train_slices = slices[slices.index.get_level_values('track_id').isin(train_tracks) & slices['pulse_no'].isin(train_pulses)]
        test_slices = slices[slices.index.get_level_values('track_id').isin(test_tracks) & slices['pulse_no'].isin(test_pulses)]
    print('done', flush=True)
    return train_slices, test_slices


class Step(AbstractStep):

    step_name = 'MIdnc'

    required_parameters = ['nearest_neighbor_k', 'entropy_estimation_correction', 'n_iters', 'train_on', 'test_set_size']
    input_files = ['extracted_slices']
    output_files = {'mutual_information': '.pkl.gz', 'mutual_informations': '.pkl.gz'}#, 'prediction_probas': '.pkl.gz'}



    def perform(self, **kwargs):
        print('------ESTIMATING MI (DISCRETE)------')

        nearest_neighbor_k = kwargs['nearest_neighbor_k']
        entropy_estimation_correction = kwargs['entropy_estimation_correction']
        n_iters = kwargs['n_iters']
        train_on = kwargs['train_on']
        assert train_on in ('same', 'other_tracks', 'other_pulses', 'other_tracks_and_pulses')
        test_set_size = kwargs['test_set_size']

        slices: pd.DataFrame = self.load_file('extracted_slices')

        # print('Reformating data', end='... ', flush=True)
        # data = list(zip(1 * (slices['target'].eq(tpos)), slices['flat_data']))#[(1 * (row['target'] == tpos), np.array(row['flat_data'])) for idx,row in slices.iterrows()]


        def get_mi(i):
            print(f'Iteration {str(i)}/{str(n_iters)}:')
            _, test_slices = split_data(slices, train_on, test_set_size=test_set_size, seed=i)
            return compute_mi(test_slices['target'].to_numpy(), np.array([np.array(y) for y in test_slices['flat_data']]), nearest_neighbor_k, entropy_estimation_correction)
            
        print('Computing MI', end='... ', flush=True)
        mutual_informations = np.array([get_mi(i) for i in range(n_iters)])
        mutual_information = np.mean(mutual_informations)
        # mutual_information = compute_mi(slices['target'].to_numpy(), np.array([np.array(y) for y in slices['flat_data']]), nearest_neighbor_k, entropy_estimation_correction)
        print('done.')
        print(f"{mutual_information=} +- {np.std(mutual_informations)}")

        self.save_file(mutual_information, 'mutual_information')
        self.save_file(mutual_informations, 'mutual_informations')




