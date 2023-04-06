# finding_vivid_tracks.py
from core.step_manager import AbstractStep
import pandas as pd

class Step(AbstractStep):


    step_name = 'VT'

    required_parameters = ['vivid_track_criteria']
    input_files = ['track_information']
    output_files = {'vivid_tracks': '.pkl.gz'}

    def perform(self, **kwargs):
        print('------PRESELECTING TRACKS ------')
        vivid_track_criteria = kwargs['vivid_track_criteria']
        track_information : pd.DataFrame =  self.load_file('track_information')

        vivid_tracks = track_information.index

        for field, method, comparator, threshold  in vivid_track_criteria:

            assert method in ('value', 'index', 'rank', 'lambda')
            assert comparator in ('gt', 'ge', 'lt', 'le', 'e')

            vivid_track_information = track_information.loc[vivid_tracks]

            if method == 'value':
                evaluate = lambda x: x[field]
            elif method == 'rank':
                evaluate = lambda x: x[field].rank(method='max')/len(x)
            elif method == 'index':
                evaluate = lambda x: pd.Series(x.index)
            elif method == 'lambda':
                evaluate = lambda x: x.pipe(eval(field))

            compare = {
                'gt': pd.Series.__gt__,
                'ge': pd.Series.__ge__,
                'lt': pd.Series.__lt__,
                'le': pd.Series.__le__,
                'e': pd.Series.__eq__,
            }[comparator]

            print('Determining vivid tracks', end='... ', flush=True)
            vivid_tracks = vivid_track_information[compare(evaluate(vivid_track_information), threshold)].index
            print('done.', flush=True)

            print(f'Vivid tracks (with {field} over {str(threshold)} using method ({method})): {str(len(vivid_tracks))} / {str(len(track_information))}')
            

        self.save_file(vivid_tracks.tolist(), 'vivid_tracks')
