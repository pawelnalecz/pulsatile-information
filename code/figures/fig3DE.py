from matplotlib import pyplot as plt
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core import experiment_manager, factory

from figures.local_config import figure_output_path
from figures.figure_settings import *
from integrity import check_and_fetch
from utils import utils

check_and_fetch.check_and_fetch_necessary()

output_path = Path(figure_output_path).absolute() / "fig34"
output_path.mkdir(parents=True, exist_ok=True)



learning = True

groups = [
    ([experiment for experiment in experiment_manager.chosen_experiments_interval if experiment in experiment_manager.best_experiments], "Interval encoding", False, False, "figS1D--interval.svg"),
    ([experiment for experiment in experiment_manager.chosen_experiments_interval_with_gap if experiment in experiment_manager.best_experiments], "Interval encoding with minimal gap", False, False, "figS1D--with_gap.svg"),
]

for group_it, (experiments, title, regular, onOtherDataSet, figname) in enumerate(groups):
    
    mutual_information = {}
    mutual_information_std = {}
    start_times = [-2]


    for experiment in experiments:

        for start_time in start_times:#range(-10,7):
            for end_time in range(max(start_time+3, 3), min(start_time+14, 11)):
                slice_length = end_time - start_time - 2
                t_pos = end_time -1
                pwms = t_pos-1 #max(t_pos-1, 0)

                parameters = {
                    **experiment_manager.default_parameters,
                    **experiment_manager.experiments[experiment],
                    'theoretical_parameters': experiment_manager.theoretical_parameters[experiment],
                    'trim_end': experiment_manager.trim_end(experiment),
                    'target_position': t_pos,
                    'slice_length': slice_length,
                    'pulse_window_matching_shift': pwms,
                    'correct_consecutive': 1, # should be 2 according to Methods
                    

                    **({
                        'correct_consecutive': 0,
                        'n_pulses': 19,
                        'pulse_length': experiment_manager.theoretical_parameters[experiment]['minutes_per_timepoint'],
                        'train_on_other_experiment': onOtherDataSet,
                        'r_slice_length': 1,
                    } if regular else {}),
                }

                if onOtherDataSet:
                    complementary_experiment = experiment_manager.get_complementary_experiment(experiment)
                    parameters1 = {
                        **parameters,
                        **experiment_manager.experiments[complementary_experiment],
                        }
                else:
                    parameters1=None

                print(parameters)
                chain = (
                    factory.compute_information_transmission(regular, learning)(parameters, parameters1)
                )
                minutes_per_timepoint = parameters['theoretical_parameters']['minutes_per_timepoint']
                empirical_measures = chain.load_file('empirical_measures')
                input_information_correction = empirical_measures['input entropy assuming independent']- empirical_measures['input entropy']
                mutual_information.update({(experiment, start_time, end_time): (chain.load_file('mutual_information') - input_information_correction) * 60 / minutes_per_timepoint})
                mutual_information_std.update({(experiment, start_time, end_time): np.std(chain.load_file('mutual_informations')) * 60 / minutes_per_timepoint})

                
    mutual_information_df = pd.Series(mutual_information, name = 'transmitted information [bit/h]')
    mutual_information_std_df = pd.Series(mutual_information_std, name = 'transmitted information std [bit/h]')
    mutual_information_df.index.names = ['experiment', 'start time', 'end time']
    mutual_information_std_df.index.names = ['experiment', 'start time', 'end time']
    
    mutual_information_aggregated_df = mutual_information_df.groupby(['start time', 'end time']).mean()
    mutual_information_std_both_errors_df: pd.Series = np.sqrt(mutual_information_df.groupby(['start time', 'end time']).std() ** 2 + (mutual_information_std_df**2).groupby(['start time', 'end time']).mean())
    mutual_information_std_both_errors_df.name = 'transmitted information std [bit/h]'

    mutual_information_stdeotm_both_errors_df: pd.Series = mutual_information_std_both_errors_df / np.sqrt(mutual_information_df.groupby(['start time', 'end time']).size())
    mutual_information_stdeotm_both_errors_df.name = 'transmitted information stdeotm [bit/h]'


    print(mutual_information_aggregated_df)
    print(mutual_information_std_both_errors_df)



    mi_df = pd.concat([mutual_information_aggregated_df, mutual_information_std_both_errors_df, mutual_information_stdeotm_both_errors_df], axis='columns')
            
        
    plt.figure('vs start time', figsize=(12, 3.5)) #(4*len(groups)+1, 3.5)
    plt.subplot(1, len(groups), group_it+1)
    def add_red(tup, red):
        print(tup, red)
        return (1-(1-tup[0]) *(1-red),) + tup[1:]
    def add_green(tup, green):
        print(tup, green)
        return tup[0:1] + (1-(1-tup[1]) *(1-green),) + tup[2:]
    
    # mutual_information_df.reset_index().plot.scatter('end time', 'transmitted information [bit/h]', ax=plt.gca())

    for (start_time, mis), pattern in zip(mi_df.groupby('start time'), ('-')*len(start_times)): #, ('-',)*5
        print(mis)
        # mis.reset_index().plot.line('end time', 'transmitted information [bit/h]', yerr='transmitted information stdeotm [bit/h]', capsize=2, ecolor='k', label=f"{-int(np.round(start_time)):d}", ax=plt.gca(), ls=pattern, color=add_green(plt.get_cmap('Blues_r')(100 if start_time == -2 else 160), 0.2*(start_time+2))) #plt.get_cmap('Blues_r')(it*23+100)
        mis.reset_index().plot.line('end time', 'transmitted information [bit/h]', label=f"{-int(np.round(start_time)):d}", ax=plt.gca(), ls=pattern, color='slateblue') #plt.get_cmap('Blues_r')(it*23+100)
    mutual_information_df.reset_index().plot.line('end time', 'transmitted information [bit/h]', ax=plt.gca(), ls='none', marker='o', ms=5, color='none', alpha=0.4, markerfacecolor='b')    

    if len(start_times) == 1:
        plt.gca().get_legend().remove()
    else:
        plt.legend(title='Earliest timepoint in window \n[min before slot]', fontsize='large')


    #overall_empiricals_per_parameter_set[(overall_empiricals_per_parameter_set.index.get_level_values('slice_length') == 5 ) & (overall_empiricals_per_parameter_set.index.get_level_values('target_position') == 4 )
    mutual_information_aggregated_df.groupby(['start time', 'end time']).mean().reindex([(-2, 5)]).reset_index().plot.scatter('end time', 'transmitted information [bit/h]', ax=plt.gca(), c='None', edgecolors='purple', zorder=2.5, s=100)
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
    plt.grid(True, ls='--')

    if len(start_times) == 1:
        plt.twiny()
        plt.xlim(xlim[0]-start_times[0]+1, xlim[1]-start_times[0]+1)
        plt.xlabel('Window length [time points]', fontsize='large')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')



    plt.savefig(output_path / 'fig3DE.svg')

    print(mi_df)
    print(mutual_information_df)



plt.show()









