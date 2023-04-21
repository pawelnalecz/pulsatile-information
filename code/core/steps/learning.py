# learning.py
from core.step_manager import AbstractStep, Chain
import pandas as pd

from utils.data_utils import split_data
from sklearn.neighbors import KNeighborsClassifier


class Step(AbstractStep):

    step_name = 'L'

    required_parameters = ['n_iters', 'train_on', 'nearest_neighbor_k', 'test_set_size', 'train_on_other_experiment']
    input_files = ['extracted_slices']
    output_files = {'prediction_results': '.pkl.gz'}


    def __init__(self, chain: Chain) -> None:
        if chain.parameters['train_on_other_experiment']:
            self.input_files = self.input_files + ['train_slices']
        super().__init__(chain)


    def perform(self, **kwargs):
        print('------LEARNING ------')
        n_iters = kwargs['n_iters']
        train_on = kwargs['train_on']
        nearest_neighbor_k = kwargs['nearest_neighbor_k']
        test_set_size = kwargs['test_set_size']
        train_on_other_experiment = kwargs['train_on_other_experiment']

        clfs = {nearest_neighbor_k: KNeighborsClassifier(n_neighbors=nearest_neighbor_k)}
        assert train_on in ('same', 'other_tracks', 'other_pulses', 'other_tracks_and_pulses')
        
        slices = self.load_file('extracted_slices')
        train_data = self.load_file('train_slices') if train_on_other_experiment else slices

        results = pd.DataFrame(columns=['track_id', 'slice_no']).set_index(['track_id', 'slice_no'])
        results_list = []
        print('Learning & computing predictions', end='...\n', flush=True)
        for i in range(n_iters):
            print(f'Iteration {str(i)}/{str(n_iters)}:')
            train_slices, test_slices = split_data(slices, train_data=train_data, train_on=train_on, test_set_size=test_set_size, seed=i)

            for clf_name, clf in clfs.items():
                print('Classifier:', clf_name, ':', end=' ', flush=True)
                print('Learning', end='... ', flush=True)
                clf.fit(train_slices['flat_data'].to_list(), train_slices['target'].to_list())
                print('Predicting', end='... ', flush=True)
                y_pred = pd.Series(clf.predict(test_slices['flat_data'].to_list()), index=test_slices.index)
                print('Saving results', end='... ', flush=True)
                partial_results = pd.DataFrame({'y_pred': y_pred, 'y_true': test_slices['target']}, index = test_slices.index)
                partial_results['classifier'] = clf_name
                partial_results['iteration'] = i
                results_list.append(partial_results)

                print('done', flush=True)
        
        results = pd.concat(results_list)
        results['pulse_no'] = slices['pulse_no']
        results = results.reset_index().set_index(['classifier', 'iteration', 'track_id', 'slice_no']).sort_index()



        self.save_file(results, 'prediction_results')

