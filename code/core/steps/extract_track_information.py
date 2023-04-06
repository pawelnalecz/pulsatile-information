# finding_good_tracks.py
import pandas as pd
from progressbar import ProgressBar

from core.step_manager import AbstractStep

class Step(AbstractStep):

    step_name = 'TI'

    required_parameters = []
    input_files = ['quantified_tracks', 'blinks']
    output_files = {'track_information': '.pkl.gz', }


    def perform(self, **kwargs):
        print('------ EXTRACTING TRACK PARAMETERS ------')

        quantified_tracks = self.load_file('quantified_tracks')
        blinks : pd.Series = self.load_file('blinks')

        if blinks[1]-blinks[0] == 1:
            blinks = blinks[90:]
            blinks.index = blinks.index-90

        track_information = pd.DataFrame({ 
            track_id: {
            'length': len(track),
            'std_Q2': track['Q2'].std(),
            'std_Q3backw': track['Q3backw'].std(),
            'mean_minus_median_Q3backw': track['Q3backw'].mean() - track['Q3backw'].median(),
            'mean_abs_dQ2': track['dQ2'].abs().mean(),
            'std_dQ2': track['dQ2'].std(),
            'mean_abs_dQ3backw': track['dQ3backw'].abs().mean(),
            'mean_abs_Q3backw_minus_1': (track['Q3backw']-1).abs().mean(),
            'std_dQ3backw': track['dQ3backw'].std(),
            'std_Q3backw_minus_1': (track['Q3backw']-1).std(),
            'local_std_dQ2': track['dQ2'].rolling(5).std().mean(),
            'local_std_dQ3backw': track['dQ3backw'].rolling(5).std().mean(),
            'smoothed_std_dQ2': track['dQ2'].rolling(5).mean().std(),
            'smoothed_std_dQ3backw': track['dQ3backw'].rolling(5).mean().std(),
            'max_minus_min_dQ3backw': (track['dQ3backw'].rolling(60).max() - track['dQ3backw'].rolling(60).min()).mean(),
            'max_minus_min_Q3backw': (track['Q3backw'].rolling(60).max() - track['Q3backw'].rolling(60).min()).mean(),
            'max_minus_min_Q2': (track['Q2'].rolling(60).max() - track['Q2'].rolling(60).min()).mean(),
            'first_blink_max_minus_min_Q3backw': (lambda x: x.max()-x.min())(track['Q3'].reindex(range(blinks[0]-30, blinks[1]))),
            'first_blink_Q2': track['Q2'][blinks[0]:blinks[0]+30].mean() - track['Q2'][blinks[0]-30:blinks[0]].mean(),
            'first_blink_Q3': track['Q3'][blinks[0]:blinks[0]+30].mean() - track['Q3'][blinks[0]-30:blinks[0]].mean(),
            #  'a_long_blink_Q2': track['Q2'][blinks[a_long_blink]:blinks[a_long_blink]+20].mean() - track['Q2'][blinks[a_long_blink]-20:blinks[a_long_blink]].mean(),
            #  'a_long_blink_Q3backw': track['Q3backw'][blinks[a_long_blink]:blinks[a_long_blink]+20].mean() - track['Q3backw'][blinks[a_long_blink]-20:blinks[a_long_blink]].mean(),
            }
            for track_id, track in enumerate(ProgressBar()(quantified_tracks))
        }).T
        track_information['global_per_local_std_dQ2'] = track_information['std_dQ2'] / track_information['local_std_dQ2']
        track_information['global_per_local_std_dQ3backw'] = track_information['std_dQ3backw'] / track_information['local_std_dQ3backw']
        track_information['std_Q3backw_minus_1_over_mean_abs_dQ3backw'] = track_information['std_Q3backw_minus_1'] - track_information['mean_abs_dQ3backw']

        self.save_file(track_information, 'track_information')

