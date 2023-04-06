# learning.py
import pandas as pd

from steps.learning import classifiers
from core.step_manager import AbstractStep

class Step(AbstractStep):

    step_name = 'Lods'

    required_parameters = ['classifiers']
    input_files = ['extracted_slices', 'train_on_dataset']
    output_files = {'prediction_results': '.pkl.gz'}

    def perform(self, **kwargs):
        print('------LEARNING ON TRAIN DATASET------')
        clfs = {clf: classifiers[clf] for clf in kwargs['classifiers']}

        train_slices : pd.DataFrame = self.load_file('train_on_dataset')
        test_slices : pd.DataFrame = self.load_file('extracted_slices')

        # results = pd.DataFrame(columns=['track_id', 'slice_no']).set_index(['track_id', 'slice_no'])
        print('Learning & computing predictions', end='...\n', flush=True)
        
        def train_and_test(i, clf_name, clf):
            print(clf_name, ':', end=' ', flush=True)
            print('Learning', end='... ', flush=True)
            clf.fit(train_slices['flat_data'].to_list(), train_slices['target'].to_list())

            print('Predicting', end='... ', flush=True)
            y_pred = pd.Series(clf.predict(test_slices['flat_data'].to_list()), index=test_slices.index)

            print('Saving results', end='... ', flush=True)
            partial_results = pd.DataFrame({'y_pred': y_pred, 'y_true': test_slices['target']}, index = test_slices.index)
            partial_results['classifier'] = clf_name
            partial_results['iteration'] = i
            print('done', flush=True)
            return partial_results
        
        results = pd.concat(train_and_test(i, clf_name, clf) for i in range(1) for clf_name, clf in clfs.items())
        results['pulse_no'] = test_slices['pulse_no']
        results = results.reset_index().set_index(['classifier', 'iteration', 'track_id', 'slice_no']).sort_index()

        self.save_file(results, 'prediction_results')

