import pandas as pd
from progressbar import ProgressBar
from typing import List
import time

from core.step_manager import AbstractStep, Chain

flatten = lambda t: [item for sublist in t for item in sublist]

def is_between(x, low, high):
    return x >= low and x < high


class Step(AbstractStep):

    step_name = 'SEregr'

    required_parameters = ['slice_length', 'fields_for_learning', 'take_tracks', 'pulse_window_matching_shift', 'trim_start', 'trim_end', 'trim_breaks_longer_than',]
    input_files = ['quantified_tracks', 'blinks', 'previous_pulse_lengths']
    output_files = {'extracted_slices': '.pkl.gz'}

    def __init__(self, chain: Chain) -> None:
        if chain.parameters['take_tracks'] == 'preselected':
            self.input_files = self.input_files + ['vivid_tracks']
        super().__init__(chain)

    def perform(self, **kwargs):
        print('------ SLICE EXTRACTION FOR REGRESSION ------')
        slice_length = kwargs['slice_length']
        take_tracks = kwargs['take_tracks']
        fields_for_learning = kwargs['fields_for_learning']
        trim_start = kwargs['trim_start']
        trim_end = kwargs['trim_end']
        trim_breaks_longer_than = kwargs['trim_breaks_longer_than']
        pwms = kwargs['pulse_window_matching_shift']
        print(take_tracks)

        quantified_tracks : List[pd.DataFrame] = self.load_file('quantified_tracks')
        blinks = list(self.load_file('blinks'))
        previous_pulse_lengths = self.load_file('previous_pulse_lengths')
        vivid_tracks = self.load_file('vivid_tracks')

        if take_tracks is None:
            tracks_to_take =  range(len(quantified_tracks))
        elif take_tracks == 'full':
            full_track_length = len(quantified_tracks[0])
            tracks_to_take = [track_no for track_no, track in enumerate(quantified_tracks) if len(track) == full_track_length ]
        elif take_tracks=='preselected':
            tracks_to_take = vivid_tracks
        else:
            tracks_to_take = take_tracks
        print('done', flush=True)


        print('Extracting slices of length', slice_length, end='...\n', flush=True)

        start = min([track.index[0] for track in quantified_tracks]) - max(0, pwms)
        end = max([track.index[-1] for track in quantified_tracks]) + 1 + max(0, -pwms)
        pulse_nos = pd.Series(
            [pd.NA] * (blinks[0]-start) 
            + flatten([[i] *(blinks[i+1] - blinks[i]) for i in range(len(blinks) - 1)])
            + [len(blinks) -1] * (end - blinks[-1])
            , index=range(start, end), dtype="Int64")
        last_blink =pd.Series([blinks[pulse_no] if not pd.isna(pulse_no) else pd.NA for pulse_no in pulse_nos], index=range(start, end), dtype="Int64")

        
        break_pulse_nos = previous_pulse_lengths[previous_pulse_lengths >= trim_breaks_longer_than].index -1
        breaks = [(blinks[br] + previous_pulse_lengths[br+2] + pwms, blinks[br+2] + pwms) for br in break_pulse_nos]
        
        trim_start = trim_start or (0, 0)
        trim_start_tp = blinks[trim_start[0]] + trim_start[1] + pwms
        trim_end_tp = blinks[trim_end[0]] + trim_end[1] + pwms if trim_end  else None

        st = time.time()
        slices_df = pd.DataFrame({
                'track_id': track_id,
                'slice_no': slice_no,
                'flat_data': the_slice.to_numpy().flatten(),
                'target': slice_no - last_blink[slice_no-pwms],
                'pulse_no': pulse_no,
            } for track_id in ProgressBar()(sorted(tracks_to_take))
                for track in (quantified_tracks[track_id],)
                    for the_slice in track[fields_for_learning].reindex(range(trim_start_tp - slice_length + 1 , trim_end_tp)).rolling(slice_length)
                        if not((len(the_slice) < slice_length) or the_slice.isnull().values.any())
                        for slice_no in (the_slice.index[-1],)
                            for pulse_no in (pulse_nos[slice_no-pwms],)
                                if not any(is_between(slice_no, *br) for br in breaks)
        ).set_index(['track_id', 'slice_no'])

        et = time.time()

        print(f"{et-st:.3f}s elapsed")
        

        print('done', flush=True)

        print(f'Extracted {len(slices_df)} slices from {len(quantified_tracks)} tracks')

        self.save_file(slices_df, 'extracted_slices')


