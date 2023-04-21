mins_per_hour = 60

def _qtracks_filename_core(directory, n_tracks):
    return directory + '--qtracks--n_' + str(n_tracks)  # selected by c.o.v.

def qtracks_cache_filename(directory, n_tracks):
    return _qtracks_filename_core(directory, n_tracks) + '.pkl.gz'


def load_shuttletracker_data(directory, use_cache=True, n_tracks=1000, exclude_tracks_with_incomplete_nuclei=False):
    import os.path
    import datetime
    import pickle, gzip
    from . import shuttletracker

    assert os.path.isdir(directory), directory

    cache_file_name = qtracks_cache_filename(directory, n_tracks)
    print('file ' + os.path.abspath(cache_file_name) + ' exists? ' +  str(os.path.exists(cache_file_name)))
    if use_cache and os.path.exists(cache_file_name):
        print(f'Loading tracks data from cache ({cache_file_name})... ', end='', flush=True)
        with gzip.open(cache_file_name, 'rb') as gzpof:
            qtracks = pickle.load(gzpof)
            print('done.')
    else:
        qtracks = dict()
        t_begin = datetime.datetime.now()
        qtracks[directory] = shuttletracker.load_tracks(directory, ntracks=n_tracks,
                                                        exclude_tracks_with_incomplete_nuclei=exclude_tracks_with_incomplete_nuclei)
        delta_t = datetime.datetime.now() - t_begin
        print('Loading and merging {} tracks took {}m{}s.'
            .format(len(qtracks[directory]), delta_t.seconds // 60, delta_t.seconds % 60))

        if use_cache:
            with gzip.open(cache_file_name, 'wb') as pof:
                pickle.dump(qtracks, pof)
    
    print('Renaming...', flush=True)
    qtracks = {
        (key[2:] if key[0:2] == './' else key): qtracks[key]
        for key in qtracks.keys()
    }

    qtracks = {
        (key[:-1] if key[-1] == '/' else key): qtracks[key]
        for key in qtracks.keys()
    }
    
    if type(qtracks)==list:
        print('WARNING: Received qtracks as a list')
        qtracks = {directory: qtracks}

    if directory not in qtracks.keys() and len(qtracks) == 1:
        print(f'WARNING: Found qtracks for directory \'{list(qtracks.keys())[0]}\', expected \'{directory}\'. Assuming this is the correct data')
        qtracks = {directory: qtracks[list(qtracks.keys())[0]]}
    qtracks_renamed = [track.rename(mapper=lambda s: s.replace('ERKTR', 'ERKKTR'), axis=1) for _,track,_ in qtracks[directory]]
    return qtracks_renamed


def extract_stimulation(directory, n_tracks=1000, export_csv=True, show_plot=True, export_plot=True,
                        time_padding=0, time_point_shift=-90,
                        plot_size=(22, 2), line_color='black', hide_spines=True, png_dpi=72,
                        subplot_adjustment={'left':0.015, 'right':0.985, 'top':0.79, 'bottom':0.37},
                        useShuttleTrackerData=True, Q=None):
    import os
    import os.path
    from glob import glob
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as tckr
    import pickle
    
    assert os.path.isdir(directory), os.path.abspath(directory)
    assert directory[-1] != '/'
    assert show_plot if export_plot else True

    if useShuttleTrackerData:
        Q = Q or load_shuttletracker_data(directory, n_tracks=n_tracks)
        Q_df = Q[0]
        print (Q_df)
        if not('img_Signal_intensity_median' in Q_df.columns or 'img_FGFR_intensity_median' in Q_df.columns):
            print(f'WARNING: Proper column not found in shuttletracker data. Using .csv data instead')
            print(list(Q_df.columns))
            useShuttleTrackerData = False
        else:
            signal_intensity_median_key = 'img_Signal_intensity_median' if 'img_Signal_intensity_median' in Q_df.columns else 'img_FGFR_intensity_median'
            print (signal_intensity_median_key, Q_df[signal_intensity_median_key])
            qq = list(Q_df[Q_df[signal_intensity_median_key]/255 > 0.5].index)

        
    if not useShuttleTrackerData:
        # read image quantification CSV files
        
        q_fns = glob(directory + os.sep + '*-img_quant.csv')
        q_fns = sorted(q_fns)
        q_fns = glob(directory + os.sep + '*-img_quant.csv')
        q_fns = sorted(q_fns)

        q = pd.read_csv(q_fns[0])

        if 'Signal_intensity_median' in q.columns or 'FGFR_intensity_median' in q.columns:
            hasExperimentalBlinksData = True
        else:
            q_fns = glob(directory + '-signal' + os.sep + '*-img_quant.csv')
            
            if len(q_fns):
                print(f'{len(q_fns)=}')
                q_fns = sorted(q_fns)
                q = pd.read_csv(q_fns[0])
                hasExperimentalBlinksData = True
            else:    
                print('WARNING: No experimental data on blinks timing found! Using design data')
                hasExperimentalBlinksData = False
                with open(directory + os.sep + 'input_sequence.pkl', 'rb') as pof:
                    qq = (lambda x: x[x==1].index)(pickle.load(pof))

        if hasExperimentalBlinksData:
            for i, q_fn in enumerate(q_fns[1:]):
                single_q = pd.read_csv(q_fn)
                q = q.append(single_q, ignore_index=True)
            q.index.name = 'time_point_index'
            
            signal_intensity_median_key = 'Signal_intensity_median' if 'Signal_intensity_median' in q.columns else 'FGFR_intensity_median'

            # prepare a concise data structure
            print(q)
            qq = list(q[ q[signal_intensity_median_key]/255 > 0.5 ].index)

    print(qq)
    qq = pd.DataFrame(data={'signal_on_timepoint_index': qq})
    if export_csv:
        fn = directory + '--stimulation.csv'
        qq.to_csv(fn)
        print(f'Stimulation time points written out to {fn}')

    if show_plot:
        # plot
        fig, ax = plt.subplots(figsize=plot_size)
        ax.plot(q.index + time_point_shift, 
                q[signal_intensity_median_key]/255, 
                color=line_color)
        
        # axes
        ax.xaxis.set_major_locator(tckr.MultipleLocator(60))
        ax.yaxis.set_major_locator(tckr.NullLocator())
        if hide_spines:
            for sp_off in ['top', 'bottom', 'right', 'left']:
                ax.spines[sp_off].set_color('none')
       
        # ranges
        x_begin = q.index[0]  - time_padding + time_point_shift
        x_end   = q.index[-1] + time_padding + time_point_shift
        x_range = (x_begin, x_end)
        ax.set_xlim(x_range)
        ax.set_ylim((0, 1.04))
        
        # labels
        ax.set_title(directory.replace('_', '\\_') if plt.rcParams['text.usetex'] 
                                                   else directory)
        ax.set_xlabel('Time [min]')
        ax.set_ylabel('Stimulation');
        plt.subplots_adjust(**subplot_adjustment)
        plt.show()
    
        if export_plot:
            fn = directory + '--stimulation.png'
            fig.savefig(fn, dpi=png_dpi)
            print(f'Stimulation time points plot saved as {fn}')
             
    n_stimulation_time_points = len(qq)
    print(f'Total stimulation time points: {n_stimulation_time_points}')
    
    stimulation_first_time_point = qq['signal_on_timepoint_index'].head(1).values[0]
    print(f'First stimulus at time point index {stimulation_first_time_point}')
    
    return (qq, stimulation_first_time_point)

