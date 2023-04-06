import pandas as pd
from progressbar import ProgressBar

from core.step_manager import AbstractStep, Chain

class Step(AbstractStep):

    step_name = 'SEregrAll'

    required_parameters = ['experiment_list']
    input_files = []
    output_files = {'extracted_slices': '.pkl.gz', 'quantified_tracks': '.pkl.gz'}


    def __init__(self, chain: Chain) -> None:
        experiment_list = chain.parameters['experiment_list']
        self.input_files = ['quantified_tracks_' + experiment for experiment in experiment_list] + ['extracted_slices_' + experiment for experiment in experiment_list]
        super().__init__(chain)

    def perform(self, **kwargs):
        print('------ SLICE EXTRACTION FOR REGRESSION FROM ALL EXPERIMENTS ------')
        experiment_list = kwargs['experiment_list']

        quantified_tracks_combined = []

        def combine_slices(it_experiment, experiment, quantified_tracks_combined): # !! Modifies quantified_tracks_combined !!

            print(f"Loading files for experiment '{experiment}'", end='... \n', flush=True)
            slices : pd.DataFrame = self.load_file('extracted_slices_' + experiment)
            quantified_tracks = self.load_file('quantified_tracks_' + experiment)

            print(f"Copying tracks for experiment '{experiment}'", end='... \n', flush=True)
            quantified_tracks_combined.extend(quantified_tracks[track_id] for track_id in ProgressBar()(slices.index.get_level_values('track_id').unique()))
            print(f"Copying slices for experiment '{experiment}'", end='... ', flush=True)
            return slices.reset_index().assign(track_id2=lambda x: x['track_id'] + 100000*it_experiment).drop(columns='track_id').assign(track_id=lambda x: x['track_id2']).set_index(['track_id', 'slice_no'])


        slices_combined = pd.concat(combine_slices(it_experiment, experiment, quantified_tracks_combined) for it_experiment,experiment in enumerate(experiment_list))
        
        
        self.save_file(quantified_tracks_combined, 'quantified_tracks')
        self.save_file(slices_combined, 'extracted_slices')


