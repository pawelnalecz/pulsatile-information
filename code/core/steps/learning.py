# learning.py
from core.step_manager import AbstractStep
import pandas as pd


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC



classifiers = { 
                'kNN (5 neighbors)': KNeighborsClassifier(n_neighbors=5, weights='distance'),  
                'kNN (10 neighbors)': KNeighborsClassifier(n_neighbors=10, weights='distance'),  
                'kNN (5 neighbors - no weights)': KNeighborsClassifier(n_neighbors=5),  
                'kNN (10 neighbors - no weights)': KNeighborsClassifier(n_neighbors=10),  
                'kNN (20 neighbors - no weights)': KNeighborsClassifier(n_neighbors=20),  
                'SVM (RBF)': SVC(),
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'Extra Trees': ExtraTreesClassifier(),
                'AdaBoost (DecTree)': AdaBoostClassifier(),
                'Perceptron (20,20)': MLPClassifier(hidden_layer_sizes=(20,20), max_iter=1000),
                'Naive Bayes': GaussianNB(),
}
    
train_test_split_params = {
    'test_size': 1./2.,
    'shuffle': True,
}

class Step(AbstractStep):

    step_name = 'L'

    required_parameters = ['n_iters', 'train_on', 'classifiers']
    input_files = ['extracted_slices']
    output_files = {'prediction_results': '.pkl.gz'}#, 'prediction_probas': '.pkl.gz'}



    def perform(self, **kwargs):
        print('------LEARNING ------')
        n_iters = kwargs['n_iters']
        train_on = kwargs['train_on']
        clfs = {clf: classifiers[clf] for clf in kwargs['classifiers']}
        assert train_on in ('same', 'other_tracks', 'other_pulses', 'other_tracks_and_pulses')

        slices = self.load_file('extracted_slices')

        results = pd.DataFrame(columns=['track_id', 'slice_no']).set_index(['track_id', 'slice_no'])
        results_list = []
        print('Learning & computing predictions', end='...\n', flush=True)
        for i in range(n_iters):
            print(f'Iteration {str(i)}/{str(n_iters)}:')
            print('Splitting train and test data', end='...', flush=True)
            if train_on == 'same':
                train_slices, test_slices = train_test_split(slices, **train_test_split_params, random_state=i)
            if train_on == 'other_pulses':
                train_pulses, test_pulses = train_test_split(slices['pulse_no'].unique(), **train_test_split_params, random_state=i)
                train_slices = slices[slices['pulse_no'].isin(train_pulses)]
                test_slices = slices[slices['pulse_no'].isin(test_pulses)]
            if train_on == 'other_tracks':
                train_tracks, test_tracks = train_test_split(slices.index.get_level_values('track_id').unique(), **train_test_split_params, random_state=i)
                train_slices = slices[slices.index.get_level_values('track_id').isin(train_tracks)]
                test_slices = slices[slices.index.get_level_values('track_id').isin(test_tracks)]
            if train_on == 'other_tracks_and_pulses':
                train_tracks, test_tracks = train_test_split(slices.index.get_level_values('track_id').unique(), **train_test_split_params, random_state=i)
                train_pulses, test_pulses = train_test_split(slices['pulse_no'].unique(), **train_test_split_params, random_state=i)
                train_slices = slices[slices.index.get_level_values('track_id').isin(train_tracks) & slices['pulse_no'].isin(train_pulses)]
                test_slices = slices[slices.index.get_level_values('track_id').isin(test_tracks) & slices['pulse_no'].isin(test_pulses)]
            print('done', flush=True)

            for clf_name, clf in clfs.items():
                print(clf_name, ':', end=' ', flush=True)
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

