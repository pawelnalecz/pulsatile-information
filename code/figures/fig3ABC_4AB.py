import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.factory import compute_information_transmission
from core import experiment_manager
from core.step_manager import printor

from figures.figure_settings import *
from figures.local_config import figure_output_path

from matplotlib.patches import Ellipse
from matplotlib.ticker import FormatStrFormatter

TICK_BEST_EXPERIMENTS_AVERAGE = True

output_path = figure_output_path / 'fig34'
output_path.mkdir(parents=True, exist_ok=True)


for learning, slice_length, tpos, figname in (
    (True,               5,    4, 'Fig3ABC'),
    (False,              7,    5, 'Fig4AB'),
):

    def get_parameters(experiment, regular, learning, **kwargs):
        return {
            **experiment_manager.default_parameters,
            **experiment_manager.experiments[experiment],
            'theoretical_parameters': experiment_manager.theoretical_parameters[experiment],
            'trim_end': experiment_manager.trim_end(experiment),
            'slice_length': slice_length,
            'target_position': tpos,
            'correct_consecutive': 2, # should be 2 according to Methods
            

            **({
                'correct_consecutive': 0,
                'n_pulses': 19,
                'pulse_length': experiment_manager.theoretical_parameters[experiment]['minutes_per_timepoint'],
                'r_slice_length': 1,
                'train_on_other_experiment': learning,#False,# should be equal to learning according to Methods
            } if regular else {}),
            **kwargs,
        }

    def get_complementary_parameters(parameters, experiment):
        complementary_experiment = experiment_manager.get_complementary_experiment(experiment)
        return {
            **parameters,
            **experiment_manager.experiments[complementary_experiment],
            'theoretical_parameters': experiment_manager.theoretical_parameters[complementary_experiment],
            'trim_end': experiment_manager.trim_end(complementary_experiment),
            }
    
    def get_chain(experiment, regular, learning):
        parameters = get_parameters(experiment, regular, learning)
        parameters1 = get_complementary_parameters(parameters, experiment) if parameters['train_on_other_experiment'] else None
        chain = compute_information_transmission(regular=regular, learning=learning)(parameters=parameters, parameters1=parameters1)
        return chain
        


    def get_mi(experiment, regular, learning, return_iterations=False):
        parameters = get_parameters(experiment, regular, learning)
        chain = get_chain(experiment, regular, learning)
        minutes_per_timepoint = parameters['theoretical_parameters']['minutes_per_timepoint']
        if return_iterations:
            return chain.load_file('mutual_information') / parameters['s_slice_length'] * 60 / minutes_per_timepoint,  [mi / parameters['s_slice_length'] * 60 / minutes_per_timepoint for mi in chain.load_file('mutual_informations')]
        return chain.load_file('mutual_information') / parameters['s_slice_length'] * 60 / minutes_per_timepoint


    def get_empirical_measures(experiment, regular, learning):
        chain = get_chain(experiment, regular, learning)
        return chain.load_file('empirical_measures')


    best_experiment_mis = {}
    best_experiment_mi_stdeotm = {}

    subplot_no = 3
    plt.figure(figname, figsize=(3*6,4))

    for subplot_it, (title, regular, has_gap, experiments, figletter) in enumerate((
        *([('Binary encoding', True, False, experiment_manager.chosen_experiments_pseudorandom, 'A')] if learning else []),
        ('Interval encoding', False, False, experiment_manager.chosen_experiments_interval, 'B' if learning else 'A'),
        ('Interval encoding with gap', False, True, experiment_manager.chosen_experiments_interval_with_gap, 'C' if learning else 'B'),
    )):
        
        ### Compute and extract measured values

        minutes_per_timepoint = pd.Series({experiment: experiment_manager.theoretical_parameters[experiment]['minutes_per_timepoint'] for experiment in experiments}, name='minutes_per_timepoint')
        
        mutual_information_s = {experiment: get_mi(experiment, regular=regular, learning=learning, return_iterations=True) for experiment in experiments}
        mutual_information = pd.Series({experiment: mutual_information_s[experiment][0] for experiment in experiments}, name='MI [bit/h]')
        mutual_information_std = pd.Series({experiment: np.std(mutual_information_s[experiment][1]) for experiment in experiments})

        empirical_measures = pd.DataFrame({experiment: printor(get_empirical_measures(experiment, regular, learning)) for experiment in experiments}).T
        empirical_measures['mean interval'] = empirical_measures['mean interval'] * minutes_per_timepoint
        empirical_measures['min interval'] = pd.Series({experiment: experiment_manager.theoretical_parameters[experiment]['min'] for experiment in experiments})
        empirical_measures['minutes per timepoint'] = minutes_per_timepoint

        empirical_measures['input entropy [bit/h]'] = empirical_measures['input entropy'] * 60 / minutes_per_timepoint 
        empirical_measures['input entropy assuming independent [bit/h]'] = empirical_measures['input entropy assuming independent'] * 60 / minutes_per_timepoint 
        empirical_measures['input entropy correction [bit/h]'] = empirical_measures['input entropy assuming independent [bit/h]'] - empirical_measures['input entropy [bit/h]']

        empirical_measures['transmitted information [bit/h]'] = mutual_information - empirical_measures['input entropy correction [bit/h]']

        print(empirical_measures)

        ### Plot figure 3ABC

        interesting_measures = empirical_measures[['min interval', 'mean interval', 'minutes per timepoint', 'transmitted information [bit/h]', 'input entropy [bit/h]']]

        plt.figure(figname)
        ax = plt.subplot(1, subplot_no, subplot_it+1)

        if regular:
            interesting_measures_aggregated = interesting_measures.groupby('minutes per timepoint').agg(['mean', 'std']).reset_index()
            # interesting_measures_aggregated.plot.scatter('minutes per timepoint', ('transmitted information [bit/h]', 'mean'), yerr=interesting_measures_aggregated[('transmitted information [bit/h]', 'std')], ax=ax, c='blue', label='MI', alpha=0.4)
            interesting_measures_aggregated.plot.line('minutes per timepoint', ('transmitted information [bit/h]', 'mean'), yerr=interesting_measures_aggregated[('transmitted information [bit/h]', 'std')].tolist(), marker='o', ms=3, ls='none', capsize=3, ax=ax, c='blue', label='MI', ecolor='k', barsabove=True)
            # interesting_measures_aggregated.plot.scatter('minutes per timepoint', ('input entropy [bit/h]', 'mean'), ax=ax, c='k', label='input information')
            plt.xlabel('Clock period [min]')
            xlim = (0,35)
            xaxis = 'minutes per timepoint'
        else:
            # interesting_measures.plot.scatter('mean interval', 'transmitted information [bit/h]', yerr=mutual_information_std.tolist(), ax=ax, c='blue', label='MI', alpha=0.4)
            interesting_measures.plot.line('mean interval', 'transmitted information [bit/h]', yerr=mutual_information_std.tolist(), marker='o', ms=3, ls='none', capsize=3, ax=ax, c='blue', label='MI',  ecolor='k', barsabove=True)
            # interesting_measures.plot.scatter('mean interval', 'input entropy [bit/h]', ax=ax, c='k', label='input information')

            if has_gap:
                for min_interval, min_interval_row in interesting_measures.groupby('min interval').max().iterrows():
                    plt.text(min_interval_row['mean interval']+.5, min_interval_row['transmitted information [bit/h]'] + .5, f"[{min_interval}]", horizontalalignment='center', fontsize='large')
            plt.xlabel('Mean interval between pulses [min]')
            xlim = (0,70)
            xaxis = 'mean interval'
            
        ylim = (0,12)

        plt.ylim(*ylim)
        plt.yticks(fontsize='large')
        plt.ylabel(('Information transmission rate [bit/h]\n' + ('reconstruction-based' if learning else 'direct estimate')) if not subplot_it else '', fontsize='large')
        
        plt.xlim(*xlim)
        plt.xticks(fontsize='large')
        ax.get_xaxis().get_label().set_fontsize('large')
        
        ax.legend().remove()
        plt.grid(True, ls='--')
        plt.title(title, fontsize='large')
        plt.subplots_adjust(bottom=0.15)

        plt.annotate(figletter, xy=(-.1, 1.13), xycoords='axes fraction', fontsize='xx-large', verticalalignment='top', fontweight='bold')


        ### Compute best experiments mean

        experiment_list = experiments if not regular else interesting_measures['minutes per timepoint'].unique().tolist()
        best_experiments = [experiment for experiment in experiment_list if experiment in (experiment_manager.best_experiments if learning else experiment_manager.best_experiments_new)]
        interesting_measures_for_best_experiments = \
            interesting_measures.reindex(best_experiments) if not regular \
            else interesting_measures.groupby('minutes per timepoint').mean().reindex(best_experiments).reset_index()
        
        best_experiment_mis.update({title: interesting_measures_for_best_experiments['transmitted information [bit/h]'].mean()})
        best_experiment_mi_stdeotm.update({title: interesting_measures_for_best_experiments['transmitted information [bit/h]'].std() / np.sqrt(len(interesting_measures_for_best_experiments))})


        ### Add ellipse, horizontal bar and label for best experiments mean

        plt.twinx()
        plt.ylim(*ylim)
        if TICK_BEST_EXPERIMENTS_AVERAGE:
            average_channel_capacity = best_experiment_mis[title]
            plt.hlines([average_channel_capacity], *xlim, ls='--', color='mediumblue', alpha=0.5)
            plt.yticks([average_channel_capacity], color='darkblue', fontsize='large')
            plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    
            left_of_the_best, right_of_the_best = interesting_measures_for_best_experiments[xaxis].agg(['min', 'max'])
            bottom_of_the_best, top_of_the_best = interesting_measures_for_best_experiments['transmitted information [bit/h]'].agg(['min', 'max'])
            plt.gca().add_patch(Ellipse(((left_of_the_best+right_of_the_best)/2, (top_of_the_best+bottom_of_the_best)/2), right_of_the_best-left_of_the_best+(8 if not regular else 3), top_of_the_best-bottom_of_the_best+1.5, facecolor=(.6,.6,.6), alpha=0.2, edgecolor='k'))
            


    print(best_experiment_mis)
    print(best_experiment_mi_stdeotm)

    plt.figure(figname)
    plt.savefig(output_path / (figname + '.svg'))
plt.show()
