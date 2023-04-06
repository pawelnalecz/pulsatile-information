from matplotlib import pyplot as plt
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core import experiment_manager, factory
from core.steps import computing_information_transmission, computing_information_transmission_full_3bit_matrix, get_timeline

from figures.local_config import figure_output_path
from figures.figure_settings import *
from integrity import check_and_fetch
from utils import utils

check_and_fetch.check_and_fetch_necessary()

output_path = Path(figure_output_path).absolute() / "figS3/automatic/"
output_path.mkdir(parents=True, exist_ok=True)


onOtherDataSet = False
onGoodTracks = False

groups = [
    ([experiment for experiment in experiment_manager.chosen_experiments_interval if experiment in experiment_manager.best_experiments], "Interval encoding", False, "figS1D--interval.svg"),
    ([experiment for experiment in experiment_manager.chosen_experiments_interval_with_gap if experiment in experiment_manager.best_experiments], "Interval encoding with minimal gap", False, "figS1D--with_gap.svg"),
]

for group_it, (experiments, title, regular, figname) in enumerate(groups):
    
    overall_empiricals = pd.DataFrame([])

    for experiment in experiments:

        start_times = [-2]
        for start_time in start_times:#range(-10,7):
            for end_time in range(max(start_time+3, 3), min(start_time+14, 11)):
                slice_length = end_time - start_time -2
                t_pos = end_time -1
                pwms = t_pos-1 #max(t_pos-1, 0)

                parameters = {
                    **experiment_manager.default_parameters,
                    **experiment_manager.experiments[experiment],
                    'theoretical_parameters': experiment_manager.theoretical_parameters[experiment],
                    'target_position': t_pos,
                    'slice_length': slice_length,
                    'pulse_window_matching_shift': pwms,
                    'trim_start' : (1,0) if experiment != 'min3_mean30' else (91,0),
                    'trim_end': (-1, int(np.floor(experiment_manager.theoretical_parameters[experiment]['min'] + experiment_manager.theoretical_parameters[experiment]['exp_mean']))),
                        
                    **({
                        'remove_first_pulses': 0,
                        'remove_break': 0,
                        'correct_consecutive': 0,
                        'remove_shorter_than': 0,
                        'yesno': True,
                        'n_pulses': 19,
                        'pulse_length': experiment_manager.theoretical_parameters[experiment]['minutes_per_timepoint'],
                        'timeline_extraction_method' :'normal',
                        'voting_range': [-1,0,1],
                        'loss_source_determination_method': 'sequential_averaged',
                        
                    } if regular else {'correct_consecutive': 2,}),
                }

                if onOtherDataSet:
                    parameters1 = {**parameters, **experiment_manager.experiments['min20_optmeanb' if not regular else utils.complementary_pseudorandom_experiment(experiment)] }
                else:
                    parameters1=parameters

                print(parameters)
                chain = (
                    # factory.do_the_analysis(parameters, parameters1, regular, onOtherDataSet, onGoodTracks, regular).
                    factory.detecting_blink_regr(parameters).step(get_timeline).step(computing_information_transmission_full_3bit_matrix)
                    if not regular else
                    factory.get_voting_timeline(parameters, parameters1, regular, onOtherDataSet, onGoodTracks, regular).step(computing_information_transmission)
                )
                overall_empirical = chain.load_file('information_overall_empirical')
                overall_empiricals = overall_empiricals.append(pd.DataFrame({ **parameters['theoretical_parameters'], 'experiment': experiment, 'slice_length':slice_length, 'target_position': t_pos, 'start_time':start_time, 'end_time': end_time, **overall_empirical.to_dict()}, index = ((slice_length, t_pos),)))
                
            
    print(overall_empiricals.set_index(['experiment', 'slice_length', 'target_position']))

    overall_empiricals_per_parameter_set = overall_empiricals.groupby(['minutes_per_timepoint', 'slice_length', 'target_position']).mean().groupby(['slice_length', 'target_position']).mean()

        
    plt.figure('vs start time', figsize=(12, 3.5)) #(4*len(groups)+1, 3.5)
    plt.subplot(1, len(groups), group_it+1)
    def add_red(tup, red):
        print(tup, red)
        return (1-(1-tup[0]) *(1-red),) + tup[1:]
    def add_green(tup, green):
        print(tup, green)
        return tup[0:1] + (1-(1-tup[1]) *(1-green),) + tup[2:]
    for it,((start_time, overall_empirical), pattern) in enumerate(zip(overall_empiricals_per_parameter_set.groupby('start_time'), ('-')*len(start_times))): #, ('-',)*5
        overall_empirical.plot('end_time', 'channel_capacity[b/h]', label=f"{-int(np.round(start_time)):d}", ax=plt.gca(), ls=pattern, color=add_green(plt.get_cmap('Blues_r')(100 if start_time == -2 else 160), 0.2*(start_time+2))) #plt.get_cmap('Blues_r')(it*23+100)
    if len(start_times) == 1:
        plt.gca().get_legend().remove()
    else:
        plt.legend(title='Earliest timepoint in window \n[min before slot]', fontsize='large')

    overall_empiricals_per_parameter_set[(overall_empiricals_per_parameter_set.index.get_level_values('slice_length') == 5 ) & (overall_empiricals_per_parameter_set.index.get_level_values('target_position') == 4 )].plot.scatter('end_time', 'channel_capacity[b/h]', ax=plt.gca(), c='None', edgecolors='purple', zorder=2.5)
    plt.ylabel('Information transmission rate [bit/h]' if not group_it else '', fontsize='large')
    plt.ylim(0,8) 

    ALIGN_LEFT = False
    if ALIGN_LEFT:
        plt.subplots_adjust(top=0.8, bottom=0.2, left=0.1875, right=.94)
        plt.annotate('D', xy=(0.13,.98), xycoords='figure fraction', fontsize='xx-large', verticalalignment='top', fontweight='bold')
        plt.annotate('E', xy=(0.565,.98), xycoords='figure fraction', fontsize='xx-large', verticalalignment='top', fontweight='bold')
    else:
        plt.subplots_adjust(top=0.8, bottom=0.2, left=0.0978, right=.85)
        plt.annotate('D', xy=(0.065,.98), xycoords='figure fraction', fontsize='xx-large', verticalalignment='top', fontweight='bold')
        plt.annotate('E', xy=(0.48,.98), xycoords='figure fraction', fontsize='xx-large', verticalalignment='top', fontweight='bold')
    
    plt.title(title)

    plt.xlabel('Time from pulse to decision [min]', fontsize='large')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')

    xlim = plt.xlim()
    plt.grid(True)

    if len(start_times) == 1:
        plt.twiny()
        plt.xlim(xlim[0]-start_times[0]+1, xlim[1]-start_times[0]+1)
        plt.xlabel('Window length [time points]', fontsize='large')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')



    plt.savefig(output_path / 'fig3D.svg')
    

plt.show()









