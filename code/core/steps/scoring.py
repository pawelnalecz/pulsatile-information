# scoring.py
import pandas as pd

from steps import learning
from core.step_manager import AbstractStep

class Step(AbstractStep):
    step_name = 'SC'

    required_parameters = []
    input_files = ['prediction_results']
    output_files = {'scores': '.pkl.gz', 'scores_per_track': '.pkl.gz', 'scores_per_pulse': '.pkl.gz'}


    def perform(self, **kwargs):
        print('------------ SCORING ------------')
        results = self.load_file('prediction_results').reindex(learning.classifiers.keys(), level='classifier')


        print('Scoring', end='... ', flush=True)
        scores = pd.concat(pd.DataFrame(prediction_scores(results.loc[ind, 'y_true'], results.loc[ind, 'y_pred']), index = pd.MultiIndex.from_arrays([[level] for level in ind], names=['classifier', 'iteration'])) for ind in results.groupby(level = ['classifier', 'iteration'], sort=False).indices)
        print('done', flush=True)


        print('Scoring per track', end='... ', flush=True)
        scores_per_track = pd.concat(pd.DataFrame(prediction_scores(res['y_true'], res['y_pred']), index = pd.MultiIndex.from_arrays([[level] for level in ind], names=['classifier', 'track_id'])) for ind,res in results.groupby(level = ['classifier', 'track_id'], sort=False))
        print('done', flush=True)

        print('Scoring per pulse', end='... ', flush=True)
        scores_per_pulse = pd.concat(pd.DataFrame(prediction_scores(res['y_true'], res['y_pred']), index = pd.MultiIndex.from_arrays([[level] for level in ind], names=['classifier', 'pulse_no'])) for ind,res in results.set_index('pulse_no', append=True).groupby(level=['classifier', 'pulse_no'], sort=False))
        print('done', flush=True)


        self.save_file(scores, 'scores')
        self.save_file(scores_per_track, 'scores_per_track')
        self.save_file(scores_per_pulse, 'scores_per_pulse')


def prediction_scores(y_true, y_pred, walltime=None):
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, \
                                fbeta_score, matthews_corrcoef, cohen_kappa_score
    from numpy import nan

    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = (conf_matrix[0], 0, 0, 0) if len(conf_matrix)==1 else conf_matrix.ravel() if len(conf_matrix)==2 else (conf_matrix[0][0], sum(conf_matrix[1:][0]), sum(conf_matrix[0][1:]), sum(conf_matrix[1:][1:]))
    total = tn + fp + fn + tp
    true_positive_rate        = tp/(tp + fn) if (tp + fn) > 0 else nan
    true_negative_rate        = tn/(tn + fp) if (tn + fp) > 0 else nan
    false_discovery_rate      = fp/(tp + fp) if (tp + fp) > 0 else nan

    return {
            # 'Accuracy':  accuracy_score(y_true, y_pred),
            # 'Precision': precision_score(y_true, y_pred) if (tp + fp) > 0 else nan,
            # 'Recall':    recall_score(y_true, y_pred),
            # 'F1':        fbeta_score(y_true, y_pred, beta=1),
            # 'F2':        fbeta_score(y_true, y_pred, beta=2),
            # 'MCC':       matthews_corrcoef(y_true, y_pred) if (tp + fp)>0 and (tp + fn)>0 and \
            #                                                   (tn + fp)>0 and (tn + fn)>0 else nan,
            # 'kappa':     cohen_kappa_score(y_true, y_pred),
            'TP':        tp * 1./ total,
            'FP':        fp * 1./ total,
            'FN':        fn * 1./ total,
            'TN':        tn * 1./ total,
            'TPR':       true_positive_rate,
            'TNR':       true_negative_rate,
            'FDR':       false_discovery_rate,
            'overall_score': true_positive_rate * (1 - false_discovery_rate),
            't':         walltime if not walltime is None else nan}

def prediction_scores_keys():
    return prediction_scores(pd.Series([0, 1]), pd.Series([1, 0]), pd.Series([2, 2])).keys()

