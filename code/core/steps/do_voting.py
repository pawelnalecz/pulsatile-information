# do_voting.py

from utils.utils import list_without
from core.step_manager import AbstractStep, Chain

class Step(AbstractStep):

    step_name = 'V'

    required_parameters = [
        'voting_range',
        'correct_consecutive',
    ]
    input_files = []
    output_files = {'binary_timeline': '.pkl.gz'}

    def __init__(self, chain: Chain) -> None:
        for shift in chain.parameters['voting_range']:
            self.input_files = self.input_files + [f"binary_timeline_{shift:d}"]
        super().__init__(chain)



    def perform(self, **kwargs):

        correct_consecutive = kwargs['correct_consecutive']
        voting_range = kwargs['voting_range']


        print('------ VOTING ------')

        def shift_slice_no(df, shift):
            df['slice_no'] = df['slice_no'] - shift
            return df
            

        shifted_binary_timelines = [self.load_file(f"binary_timeline_{shift:d}").reset_index('slice_no').pipe(shift_slice_no, shift).set_index('slice_no', append=True) for shift in voting_range]

        binary_timeline = sum(shifted_binary_timelines).dropna().pipe(lambda x: 1*(x >= 2))
        print(set(shifted_binary_timelines[1].index.get_level_values('track_id').unique()) ^ set(shifted_binary_timelines[0].index.get_level_values( 'track_id').unique()))
        print(set(shifted_binary_timelines[2].index.get_level_values('track_id').unique()) ^ set(shifted_binary_timelines[1].index.get_level_values( 'track_id').unique()))
        print(set(shifted_binary_timelines[0].index.get_level_values('track_id').unique()) ^ set(shifted_binary_timelines[2].index.get_level_values( 'track_id').unique()))
        print(shifted_binary_timelines)
        print(sum([sbt * 0 + 1 for sbt in shifted_binary_timelines]).dropna().pipe(lambda x: x[~(x['input_blinks'] == 3)]))
        print(sum([sbt * 0 + 1 for sbt in shifted_binary_timelines]))
        print(binary_timeline)

        print('averaging', end=' ...', flush=True)

   
        if correct_consecutive:
            print('correcting consecutive', end=' ...', flush=True)
            for i in range(correct_consecutive):
                binary_timeline  = binary_timeline * (1- binary_timeline.groupby(list_without(binary_timeline.index.names, 'slice_no')).shift(i+1, fill_value=0))


        print('done.', flush=True)

        self.save_file(binary_timeline, 'binary_timeline')

