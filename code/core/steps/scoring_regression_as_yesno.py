# scoring.py

import pandas as pd

from steps import scoring, learning
from core.step_manager import AbstractStep

class Step(AbstractStep):

    step_name = 'SCregrYn'

    required_parameters = ['target_position']
    input_files = ['prediction_results']
    output_files = {'scores': '.pkl.gz', 'scores_per_track': '.pkl.gz', 'scores_per_pulse': '.pkl.gz'}

    def perform(self, **kwargs):
        target_position = kwargs['target_position']

        print('------------ SCORING REGRESSION AS YESNO ------------')
        results = self.load_file('prediction_results').reindex(learning.classifiers.keys(), level='classifier')

        scores, scores_per_track, scores_per_pulse = score_tracks(results, target_position)

        with pd.option_context('display.float_format', '{:.2f}'.format):
            print(scores)

        self.save_file(scores, 'scores')
        self.save_file(scores_per_track, 'scores_per_track')
        self.save_file(scores_per_pulse, 'scores_per_pulse')


def score_tracks(results, target_position):

    results = results.reindex(learning.classifiers.keys(), level='classifier')
    results['y_pred'] = results['y_pred'] == target_position
    results['y_true'] = results['y_true'] == target_position


    print('Scoring', end='... ', flush=True)
    scores = pd.concat(
        pd.DataFrame(scoring.prediction_scores(res['y_true'], res['y_pred']), index = pd.MultiIndex.from_arrays([[level] for level in ind], names=['classifier', 'iteration']))
            for ind, res in results.groupby(level = ['classifier', 'iteration'], sort=False)
    )
    print('done', flush=True)

    print('Scoring per track', end='... ', flush=True)
    scores_per_track = pd.concat(
        pd.DataFrame(scoring.prediction_scores(res['y_true'], res['y_pred']), index = pd.MultiIndex.from_arrays([[level] for level in ind], names=['classifier', 'track_id']))
            for ind,res in results.groupby(level = ['classifier', 'track_id'], sort=False)
    )
    print('done', flush=True)

    print('Scoring per pulse', end='... ', flush=True)
    scores_per_pulse = pd.concat(
        pd.DataFrame(scoring.prediction_scores(res['y_true'], res['y_pred']), index = pd.MultiIndex.from_arrays([[level] for level in ind], names=['classifier', 'pulse_no']))
            for ind,res in results.set_index('pulse_no', append=True).groupby(level=['classifier', 'pulse_no'], sort=False)
    )
    print('done', flush=True)

    return scores, scores_per_track, scores_per_pulse
