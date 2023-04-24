from sklearn.neighbors import NearestNeighbors
import numpy as np

from core.step_manager import AbstractStep
from utils.math_utils import plogp
import pandas as pd

from utils.data_utils import split_data
from core.step_manager import Chain



def compute_mi_naive(X_train: pd.Series, Y_train: pd.Series, Y_test: pd.Series, k: int = 20, correction=True) -> float:

    X_arr = X_train.to_numpy()
    Y_arr = np.array([np.array(y) for y in Y_train])
    Y_test_arr = np.array([np.array(y) for y in Y_test])


    classifier = NearestNeighbors(n_neighbors=k)
    classifier.fit(Y_arr)

    print('Finding neighbors...', end='', flush=True)
    dists, neighs = classifier.kneighbors(Y_test_arr)
    # Miller-Madow correction in bits, not nuts
    hinvln2 = 1/np.log(2)/2 if correction else 0 
    print('computing conditional entropy...', end='', flush=True)
    conditional_entropy = (
        np.sum([
            np.sum(
                plogp(
                    np.unique([X_arr[j] for j in js], return_counts=True)[1]
                ) + hinvln2
            ) - hinvln2
            for js in neighs]
        ) / (k * len(X_arr)) + np.log2(k)
    )
    print('computed.', flush=True)
    input_entropy = (np.sum(plogp(np.unique(X_arr, return_counts=True)[1]) + hinvln2) - hinvln2 - plogp(len(X_arr))) / len(X_arr)
    print(input_entropy, conditional_entropy, input_entropy - conditional_entropy)
    return input_entropy - conditional_entropy

class Step(AbstractStep):

    step_name = 'MId'

    required_parameters = ['nearest_neighbor_k', 'entropy_estimation_method', 'n_iters', 'train_on', 'test_set_size', 'train_on_other_experiment']
    input_files = ['extracted_slices']
    output_files = {'mutual_information': '.pkl.gz', 'mutual_informations': '.pkl.gz'}#, 'prediction_probas': '.pkl.gz'}

    def __init__(self, chain: Chain) -> None:
        if chain.parameters['train_on_other_experiment']:
            self.input_files = self.input_files + ['train_slices']
        super().__init__(chain)



    def perform(self, **kwargs):
        print('------ESTIMATING MI (DISCRETE)------')

        nearest_neighbor_k = kwargs['nearest_neighbor_k']
        entropy_estimation_method = kwargs['entropy_estimation_method']
        n_iters = kwargs['n_iters']
        train_on = kwargs['train_on']
        test_set_size = kwargs['test_set_size']
        train_on_other_experiment = kwargs['train_on_other_experiment']

        assert entropy_estimation_method in ('naive', 'naive_with_MM_correction', 'KSG_cce')
        assert train_on in ('same', 'other_tracks', 'other_pulses', 'other_tracks_and_pulses')

        slices: pd.DataFrame = self.load_file('extracted_slices')
        train_dataset: pd.DataFrame = self.load_file('train_slices') if train_on_other_experiment else slices


        if entropy_estimation_method in ('naive', 'naive_with_MM_correction'):
            def get_mi(i):
                print(f'Iteration {str(i)}/{str(n_iters)}:')
                _, test_slices = split_data(slices, train_dataset, train_on, test_set_size=test_set_size, seed=i) # could separate train and test data, should give statistically equal results, keeping no separation for consistency with Methods
                return compute_mi_naive(test_slices['target'], test_slices['flat_data'], test_slices['flat_data'], nearest_neighbor_k, correction=(entropy_estimation_method == 'naive_with_MM_correction'))
        elif entropy_estimation_method == 'KSG_cce':
            from cce import WeightedKraskovEstimator
            def get_mi(i):
                print(f'Iteration {str(i)}/{str(n_iters)}:')
                _, test_slices = split_data(slices, train_dataset, train_on, test_set_size=test_set_size, seed=i)
                print('Reformating data', end='... ', flush=True)
                data = list(zip(1 * (test_slices['target']), test_slices['flat_data']))#[(1 * (row['target'] == tpos), np.array(r ow['flat_data'])) for idx,row in slices.iterrows()]
                print('Computing mi', end='... ', flush=True)
                return WeightedKraskovEstimator(data).calculate_mi(k=nearest_neighbor_k) # This method cannot separate train and test datasets
        else:
            raise AssertionError(f"{entropy_estimation_method = } should be one of ('naive', 'naive_with_MM_correction', 'KSG_cce')")


            
        print('Computing MI', end='... ', flush=True)
        mutual_informations = np.array([get_mi(i) for i in range(n_iters)])
        mutual_information = np.mean(mutual_informations)
        print('done.')
        print(f"{mutual_information=} +- {np.std(mutual_informations)}")

        self.save_file(mutual_information, 'mutual_information')
        self.save_file(mutual_informations, 'mutual_informations')




