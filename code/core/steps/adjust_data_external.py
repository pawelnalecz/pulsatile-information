import pandas as pd
from pathlib import Path

from core.step_manager import AbstractStep

class Step(AbstractStep):

    step_name = 'ADext'
    
    required_parameters = ['directory', 'n_tracks']
    input_files = []
    output_files = {'blinks': '.pkl.gz', 'raw_tracks': '.pkl.gz', 'raw_tracks_df': '.pkl.gz', }


    def perform(self, **kwargs):
        print('------ IMPORTING DATA FROM CSV FILES ------')
        directory = kwargs['directory']
        n_tracks = kwargs['n_tracks']
        fields = ['nuc_area', 'nuc_center_x', 'nuc_center_y', 'nuc_H2B_intensity_mean', 'nuc_ERKKTR_intensity_mean', 'img_ERKKTR_intensity_mean', 'img_H2B_intensity_mean']
        
        print(f"Loading csv data from '{Path(directory + str(n_tracks) + '.csv').absolute()}'", end='... ', flush=True)
        external_data = pd.read_csv(directory + str(n_tracks) + '.csv')
        raw_tracks = [track[fields] for track_id,track in external_data.rename(columns={'time_in_minutes': 'time_point_index'}).set_index(['time_point_index']).groupby('track_id')]
        print('Exctacting stimulation', end='... ', flush=True)
        blinks = pd.Series(sorted(external_data[external_data['is_light_pulse'] == 1]['time_in_minutes'].unique()), name='signal_on_timepoint_index')
        print('done', flush=True)
            
        self.save_file(blinks, 'blinks')
        self.save_file(raw_tracks, 'raw_tracks')
        self.save_file(pd.concat(raw_tracks, names=['track_id'], keys=range(len(raw_tracks))), 'raw_tracks_df')
            
