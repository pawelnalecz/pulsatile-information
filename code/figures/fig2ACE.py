from matplotlib import pyplot as plt
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from figures.local_config import figure_output_path
from figures.figure_settings import *

from core import experiment_manager, factory
from core.steps import get_timeline
from matplotlib.ticker import MultipleLocator
from integrity import check_and_fetch

check_and_fetch.check_and_fetch_necessary()

output_path = Path(figure_output_path).absolute() / "fig234/automatic/"
output_path.mkdir(parents=True, exist_ok=True)



for experiment, start_pulse, end_pulse, regular, filename, figletter in ( 
    ('min3_mean30', 96, 116,  False, 'fig2C.svg', 'C'),
    ('min20_optmean', 6, 22,  False, 'fig2E.svg', 'E'),
    ('pseudorandom_pos01_period10_new', 0, 18,  True, 'fig2A.svg', 'A'),
):

    t_pos=4

    minutes_per_timepoint = experiment_manager.theoretical_parameters[experiment]['minutes_per_timepoint']

    parameters = {
        **experiment_manager.default_parameters,
        **experiment_manager.experiments[experiment],
        'theoretical_parameters': experiment_manager.theoretical_parameters[experiment],
        'target_position': t_pos,
        'trim_start' : (1,0) if experiment != 'min3_mean30' else (91,0),
        'trim_end': (-1, int(np.floor(experiment_manager.theoretical_parameters[experiment]['min'] + experiment_manager.theoretical_parameters[experiment]['exp_mean']))),

        **({
            'remove_first_pulses': 0,
            'remove_break': 0,
            'correct_consecutive': 0,
            'remove_shorter_than': 19,
            'yesno': True,
            'n_pulses': 19,
            'pulse_length': minutes_per_timepoint,
            'timeline_extraction_method' :'normal',
            'voting_range': [-1, 0, 1],
            'loss_source_determination_method': 'sequential_averaged',
        } if regular else {}),
    }

    chain = factory.detecting_blink_regr(parameters).step(get_timeline) if not regular else factory.get_voting_timeline(parameters, parameters, regular, False, False, regular)

    blinks : pd.Series = chain.load_file('blinks')
    quantified_tracks = chain.load_file('quantified_tracks')
    results = chain.load_file('prediction_results')
    previous_pulse_lengths = chain.load_file('previous_pulse_lengths')
    binary_timeline = chain.load_file('binary_timeline')
    potential_blinks = blinks if not regular else pd.Series(binary_timeline.index.get_level_values('slice_no').unique() - t_pos)
    print(blinks)
    

    xlim  = (potential_blinks[start_pulse]-5, potential_blinks[end_pulse]+20)

    plt.figure(figsize=(9,3.5))
    Ns=range(40,50) if experiment != 'min3_mean30_new' else range(50,60)
    for n in Ns:
        (1-quantified_tracks[n]['Q3backw'])[xlim[0]:xlim[1]].plot(alpha=0.3)
    for n in Ns:
        (1-quantified_tracks[n]['Q3backw'])[(lambda x: x[x>=0.5])(binary_timeline[binary_timeline.index.get_level_values('track_id') == n].groupby('slice_no').mean()['output_detections'].loc[xlim[0]:xlim[1]]).index -t_pos].plot(lw=0, marker='.')
        

    yticks = np.array([-0.2, 0, 0.2, 0.4, 0.6])
    ylim = [-0.3, 0.8]
    plt.xlim(xlim)
    plt.xticks(fontsize='large')
    plt.gca().xaxis.set_major_locator(MultipleLocator(100))
    plt.yticks(yticks, fontsize='large')
    plt.ylabel('ERK KTR translocation [a.u.]', fontsize='large') 
    plt.ylim(ylim)
    plt.xlabel('time [min]', fontsize='large')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set(linewidth=.1)
    

    ax2 = plt.gca().twinx()
    
    print(results.groupby('slice_no').mean())
    print(binary_timeline.groupby('slice_no').mean())

    def plot_percentage(df: pd.DataFrame, **kwargs):
        plotter = lambda x, color: plt.bar(x.index.get_level_values('slice_no') - t_pos, x['output_detections'], color=color, alpha=0.8, **kwargs)
        # plotter(df[(df.index.get_level_values('slice_no') -t_pos-2).isin(blinks)], 'tan')
        # plotter(df[(df.index.get_level_values('slice_no') -t_pos-1).isin(blinks)], 'darkgoldenrod')
        plotter(df[(df.index.get_level_values('slice_no') -t_pos  ).isin(blinks)], 'green')
        # plotter(df[(df.index.get_level_values('slice_no') -t_pos+1).isin(blinks)], 'orange')
        # plotter(df[(df.index.get_level_values('slice_no') -t_pos+2).isin(blinks)], 'wheat')
        plotter(df[~(
            # (df.index.get_level_values('slice_no') -t_pos-2).isin(blinks)
            # | (df.index.get_level_values('slice_no') -t_pos-1).isin(blinks)
            (df.index.get_level_values('slice_no') -t_pos  ).isin(blinks)
            # | (df.index.get_level_values('slice_no') -t_pos+1).isin(blinks)
            # | (df.index.get_level_values('slice_no') -t_pos+2).isin(blinks)
            )], 'grey')
        return df

    if regular:
        (lambda x: ax2.bar(x.index.get_level_values('slice_no') - t_pos, x['output_detections'], color='grey', alpha=0.8))(binary_timeline[binary_timeline.index.get_level_values('slice_no').isin(range(xlim[0]+t_pos, xlim[1]+t_pos))].groupby(['slice_no']).mean())
        (lambda x: ax2.bar(x.index.get_level_values('slice_no') - t_pos, x['output_detections'], color='green', alpha=1))(binary_timeline[binary_timeline.index.get_level_values('slice_no').isin(range(xlim[0]+t_pos, xlim[1]+t_pos)) & binary_timeline.index.get_level_values('slice_no').isin(blinks+t_pos)].groupby(['slice_no']).mean())
    else:
        plot_percentage(binary_timeline[binary_timeline.index.get_level_values('slice_no').isin(range(xlim[0]+t_pos, xlim[1]+t_pos))].groupby(['slice_no']).mean())



    for slice_no, row in (lambda x: x[x>-0.09] if not regular else x)(binary_timeline[(binary_timeline.index.get_level_values('slice_no').isin(np.array(range(*xlim))+t_pos))].groupby(['slice_no']).mean()).iterrows():
        horizontalalignment = 'right' if (experiment, slice_no - t_pos) in (('min3_mean30',409),) else 'left' if (experiment, slice_no - t_pos) in (('min3_mean30',434), ('min3_mean30',520), ('min3_mean30',538),) else 'center' if (slice_no - t_pos in blinks.to_list()) or ((experiment, slice_no - t_pos) in (('min3_mean30',508),)) else 'center'  # 'center' if not((experiment, slice_no - t_pos) in (('min3_mean30' ,519), ('min3_mean30' ,520), ('min3_mean30' ,538))) else 'left' #if slice_no-t_pos+1 in blinks.tolist() else 'left'
        verticalalignment = 'bottom'
        color = 'green' if (slice_no - t_pos in blinks.to_list()) else 'grey' if regular else 'none'
        rotation = 0
        ax2.annotate(f"{100*row['output_detections']:.0f}%", (slice_no-t_pos, row['output_detections']), horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, rotation=rotation, fontsize='small', color=color)


    plt.plot(*zip(*[ (x, 1.1 - 0.01) for x in blinks[blinks.isin(range(xlim[0], xlim[1]))]]), ls='none', marker='v', color='dodgerblue', ms='6')
    for blink,previous_pulse_length in zip(potential_blinks, previous_pulse_lengths if not regular else [np.nan] + [minutes_per_timepoint]*(len(potential_blinks)-1)):
        if blink < xlim[0] or blink >= xlim[1]:
            continue
        plt.annotate(f'{previous_pulse_length:.0f}', (blink-previous_pulse_length/2, 1.1 + 0.02), fontweight='bold', horizontalalignment='center', verticalalignment='bottom', fontsize='small')
        plt.plot(potential_blinks, [1.1+0.02]*len(potential_blinks), c='k', marker='|', lw=0.3, ms=2)
   

    plt.ylim(0,1.15)
    plt.ylabel('Pulse detection [% of cells]', fontsize='large', color='k')
    plt.yticks([0, 0.25, .5, .75, 1], ['0%', '25%', '50%', '75%', '100%', ], fontsize='large', color='k')
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.subplots_adjust(right=0.85, bottom=0.25)

    plt.annotate(figletter, xy=(0.02,.98), xycoords='figure fraction', fontsize='xx-large', verticalalignment='top', fontweight='bold')
    plt.savefig(output_path / filename)

    plt.figure(figsize=(10,2.5))
    plt.vlines(blinks[blinks.isin(range(xlim[0],xlim[1]))], 0, 1)
    for blink,previous_pulse_length in zip(potential_blinks, previous_pulse_lengths if not regular else [np.nan] + [minutes_per_timepoint]*(len(potential_blinks)-1)):
        if blink < xlim[0] or blink >= xlim[1]:
            continue
        plt.annotate(f'{previous_pulse_length:.0f}', (blink-previous_pulse_length/2,0.6 + 0.02), fontweight='bold', horizontalalignment='center', verticalalignment='bottom', fontsize='small')
        plt.plot(potential_blinks, [0.6+0.02]*len(potential_blinks), c='k', marker='|', lw=0.3, ms=2)
    plt.xlim(xlim)
    plt.xticks(fontsize='large')
    plt.xlabel('time [min]', fontsize='large')
    plt.ylim((0,1.25))
    plt.yticks([])
    plt.ylabel('Light stimulation', fontsize='large')
    plt.subplots_adjust(bottom=0.25)
    
    plt.savefig(output_path / (filename[:-4] + '-blinks' + filename[-4:]))




    
plt.show()

