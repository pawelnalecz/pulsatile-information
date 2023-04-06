from core.step_manager import AbstractStep
import pandas as pd

from progressbar import ProgressBar

# import loess # doi: 10.1093/mnras/stt644
# from statsmodels.tsa.seasonal import STL


class Step(AbstractStep):

    step_name = 'Q'
    
    required_parameters = []
    input_files = ['blinks', 'raw_tracks']
    output_files = {'quantified_tracks': '.pkl.gz', 'track_info': '.pkl.gz', 'isBlink': '.pkl.gz'}


    def perform(self, **kwargs):
        print('------ QUANTIFICATION ------')

        Q = self.load_file('raw_tracks') #pulses.load_shuttletracker_data(directory, n_tracks=n_tracks)
        blinks = self.load_file('blinks')

    
        blinks_set = set(blinks)
        isBlink = pd.Series([(i in blinks_set)*1 for i in Q[0].index], name='isBlink')

        print('Computing quantifications', end='... \n', flush=True)
        quantified_tracks = [make_quantification_df(traj).join(isBlink) for traj in ProgressBar()(Q)]
        print('done', flush=True)

        print('Exctracting track info', end='... ', flush=True)
        track_infos = pd.DataFrame([{**traj[['nuc_area', 'nuc_H2B_intensity_mean', 'nuc_ERKKTR_intensity_mean']].mean().to_dict(), 'start': traj.index[0], 'end': traj.index[-1] + 1} for traj in Q])
        print(track_infos)
        print('done', flush=True)
        
        print(f'Quantified {len(quantified_tracks)} tracks')

        self.save_file(quantified_tracks, 'quantified_tracks')
        self.save_file(track_infos, 'track_info')
        self.save_file(isBlink, 'isBlink')
            



def make_quantification_df(q: pd.DataFrame, time_points=None, q3_half_radius=60, loess_period=120):
    if time_points == None:
        time_points = q.index
    track_df = pd.DataFrame()
    track_df['Q1'] = -q['nuc_ERKKTR_intensity_mean']  
    track_df['Q2'] = -q['nuc_ERKKTR_intensity_mean']/q['img_ERKKTR_intensity_mean']
    track_df['Q3'] = track_df['Q2'] / track_df['Q2'].rolling(2 * q3_half_radius).mean().shift(-q3_half_radius)
    track_df['Q4'] = track_df['Q2'] / track_df['Q2'].mean()
    track_df['Q3backw'] = track_df['Q2'] / track_df['Q2'].rolling(2 * q3_half_radius, min_periods=1).mean()
    track_df['Q5'] = (lambda x: x - 5*x.diff().diff().shift(-1).rolling(5, center=True, win_type='gaussian').mean(std=2) - 30*x.diff().diff().shift(-1).rolling(5, center=True, win_type='gaussian').mean(std=2) * (x+1))(track_df['Q3backw'])
    track_df['dQ2'] = track_df['Q2'].diff()
    track_df['dQ3'] = track_df['Q3'].diff()
    track_df['dQ4'] = track_df['Q4'].diff()
    track_df['dQ3backw'] =track_df['Q3backw'].diff()
    track_df['dQ5'] = track_df['Q5'].diff()
    track_df['dSQ2'] = (track_df['Q2'] - track_df['Q2'].shift(1)) / (track_df['Q2'] + track_df['Q2'].shift(1))
    # track_df['LOESS'] = track_df['Q2'] - STL(track_df['Q2'], period=loess_period).fit().trend
    return track_df

