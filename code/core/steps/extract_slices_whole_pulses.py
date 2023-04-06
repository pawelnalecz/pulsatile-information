import pandas as pd
from progressbar import ProgressBar

from core.step_manager import AbstractStep, Chain


flatten = lambda t: [item for sublist in t for item in sublist]

class Step(AbstractStep):
    step_name = 'SEwhole'

    required_parameters = ['slice_length', 'fields_for_learning', 'take_tracks', 'target_position', 'experiment_onset', 'pulse_length', 'n_pulses']
    input_files = ['quantified_tracks', 'blinks']
    output_files = {'extracted_slices': '.pkl.gz'}


    def __init__(self, chain: Chain) -> None:
        if chain.parameters['take_tracks'] == 'preselected':
            self.input_files = self.input_files + ['vivid_tracks']
        super().__init__(chain)

    def perform(self, **kwargs):
        print('------ CUTTING INTO PULSES ------')
        slice_length = kwargs['slice_length']
        take_tracks = kwargs['take_tracks']
        fields_for_learning = kwargs['fields_for_learning']
        target_position = kwargs['target_position'] 
        experiment_onset = kwargs['experiment_onset']
        pulse_length = kwargs['pulse_length']
        n_pulses = kwargs['n_pulses']
        print(take_tracks)

        quantified_tracks = self.load_file('quantified_tracks')
        blinks : pd.Series = self.load_file('blinks')
        vivid_tracks = self.load_file('vivid_tracks')

        if take_tracks is None:
            quantified_tracks =  range(len(quantified_tracks))
        elif take_tracks == 'full':
            full_track_length = len(quantified_tracks[0])
            tracks_to_take = [track_no for track_no, track in enumerate(quantified_tracks) if len(track) == full_track_length ]
        elif take_tracks=='preselected':
            tracks_to_take = vivid_tracks
        else:
            tracks_to_take = take_tracks
        print('done', flush=True)



        if experiment_onset == None: experiment_onset = blinks[0]


        print('Extracting slices of length', slice_length, end='...\n', flush=True)

        slices_df = pd.DataFrame(
            {
            'track_id': track_id,
            'slice_no': the_slice.index[-1],
            'flat_data': the_slice.to_numpy().flatten(),
            'target':   1*(the_pulse_onset in blinks.tolist()),
            'pulse_no': pulse_no,
        } for track_id in ProgressBar()(sorted(tracks_to_take))
            for track in (quantified_tracks[track_id],)
                for pulse_no in range(n_pulses)
                    for the_pulse_onset in [experiment_onset + pulse_no * pulse_length]
                        for the_slice in [track[fields_for_learning].reindex(range(the_pulse_onset + target_position - slice_length + 1, the_pulse_onset + target_position + 1))]
                            if not((len(the_slice) < slice_length) or the_slice.isnull().values.any())
        
        ).set_index(['track_id', 'slice_no'])
        

        print('done', flush=True)

        print(f'Extracted {len(slices_df)} slices from {len(quantified_tracks)} tracks')

        self.save_file(slices_df, 'extracted_slices')



