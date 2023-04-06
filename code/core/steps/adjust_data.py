import pandas as pd

from core.step_manager import AbstractStep
from core.steps.MK import pulses


class Step(AbstractStep):

    step_name = 'AD'
    
    required_parameters = ['directory', 'n_tracks']
    input_files = []
    output_files = {'blinks': '.pkl.gz', 'raw_tracks': '.pkl.gz', 'raw_tracks_df': '.pkl.gz', }


    def perform(self, **kwargs):
        print('------ IMPORTING DATA ------')
        directory = kwargs['directory']
        n_tracks = kwargs['n_tracks']

        fields = ['nuc_area', 'nuc_center_x', 'nuc_center_y', 'nuc_H2B_intensity_mean', 'nuc_ERKKTR_intensity_mean', 'img_ERKKTR_intensity_mean', 'img_H2B_intensity_mean']
        
        print('Loading ShuttleTracker data', end='... ', flush=True)
        Q = pulses.load_shuttletracker_data(directory, n_tracks=n_tracks)
        raw_tracks = [track[fields] for track in Q]
        print('Exctacting stimulation', end='... ', flush=True)
        blinks = pulses.extract_stimulation(directory, n_tracks, export_csv=False, show_plot=False, export_plot=False, Q=Q)[0]['signal_on_timepoint_index']
        print('done', flush=True)
            
        self.save_file(blinks, 'blinks')
        self.save_file(raw_tracks, 'raw_tracks')
        self.save_file(pd.concat(raw_tracks, names=['track_id'], keys=range(len(raw_tracks))), 'raw_tracks_df')
            

