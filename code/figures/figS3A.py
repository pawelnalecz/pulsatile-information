import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core import experiment_manager, factory

from figures.local_config import figure_output_path
from figures.figure_settings import *
from integrity import check_and_fetch

from matplotlib import patches
from matplotlib.ticker import FuncFormatter

check_and_fetch.check_and_fetch_necessary()

output_path = Path(figure_output_path).absolute() / "figS3"
output_path.mkdir(parents=True, exist_ok=True)

bottom_of_the_best = +np.inf
top_of_the_best = -np.inf

learning = True

for group_it,(title, figname, regular, onOtherDataSet, experiments, additional_parameters) in enumerate((

    ('Interval encoding with a minimal gap', 'fig4B.svg', False, False,  
           [experiment for experiment in experiment_manager.chosen_experiments_interval_with_gap if experiment in experiment_manager.best_experiments],
        {},
    ),
    ('Interval encoding', 'fig3B.svg', False, False,
        [experiment for experiment in experiment_manager.chosen_experiments_interval if experiment in experiment_manager.best_experiments],

        {},
    ),
    ('Binary encoding', 'fig2B.svg', True, True, # onOtherDataSet should be True according to Methods
        [experiment for experiment in experiment_manager.chosen_experiments_pseudorandom if experiment in experiment_manager.best_experiments],
        {},
    ),

)):


    def get_parameters(experiment, regular, onOtherDataSet, vivid_track_offset):
        return {
            **experiment_manager.default_parameters,
            **experiment_manager.experiments[experiment],
            'theoretical_parameters': experiment_manager.theoretical_parameters[experiment],
            'trim_end': experiment_manager.trim_end(experiment),
            'correct_consecutive': 2,

            **({
                'correct_consecutive': 0,
                'n_pulses': 19,
                'pulse_length': experiment_manager.theoretical_parameters[experiment]['minutes_per_timepoint'],
                'r_slice_length': 1,
                'train_on_other_experiment': onOtherDataSet,
            } if regular else {}),
            
            'vivid_track_criteria': [
                ('', 'index', 'lt', 500),
                ('std_dQ2', 'rank', 'gt', vivid_track_offset),
            ],
            **additional_parameters,
        }



    def get_complementary_parameters(parameters, experiment):
        complementary_experiment = experiment_manager.get_complementary_experiment(experiment)
        return {
            **parameters,
            **experiment_manager.experiments[complementary_experiment],
            'theoretical_parameters': experiment_manager.theoretical_parameters[complementary_experiment],
            'trim_end': experiment_manager.trim_end(complementary_experiment),
            }
    
    def get_chain(experiment, regular, onOtherDataSet, vivid_track_offset):
        parameters = get_parameters(experiment, regular, onOtherDataSet, vivid_track_offset=vivid_track_offset)
        parameters1 = get_complementary_parameters(parameters, experiment) if parameters['train_on_other_experiment'] else None
        chain = factory.compute_information_transmission(regular=regular, learning=learning)(parameters=parameters, parameters1=parameters1)
        return chain
        


    def get_mi(experiment, regular, onOtherDataSet, vivid_track_offset, return_iterations=False):
        parameters = get_parameters(experiment, regular, onOtherDataSet, vivid_track_offset)
        chain = get_chain(experiment, regular, onOtherDataSet=onOtherDataSet, vivid_track_offset=vivid_track_offset)
        minutes_per_timepoint = parameters['theoretical_parameters']['minutes_per_timepoint']
        if return_iterations:
            return chain.load_file('mutual_information') / parameters['s_slice_length'] * 60 / minutes_per_timepoint,  [mi / parameters['s_slice_length'] * 60 / minutes_per_timepoint for mi in chain.load_file('mutual_informations')]
        return chain.load_file('mutual_information') / parameters['s_slice_length'] * 60 / minutes_per_timepoint


    def get_empirical_measures(experiment, regular, onOtherDataSet, vivid_track_offset):
        chain = get_chain(experiment, regular, onOtherDataSet=onOtherDataSet, vivid_track_offset=vivid_track_offset)
        return chain.load_file('empirical_measures')


    def get_average_mi_for_group(experiments, regular, onOtherDataSet, vivid_track_offset):

        minutes_per_timepoint = pd.Series({experiment: experiment_manager.theoretical_parameters[experiment]['minutes_per_timepoint'] for experiment in experiments}, name='minutes_per_timepoint')
        
        mutual_information = pd.Series({experiment: get_mi(experiment, regular=regular, onOtherDataSet=onOtherDataSet, vivid_track_offset=vivid_track_offset) for experiment in experiments}, name='MI [bit/h]')

        empirical_measures = pd.DataFrame({experiment: get_empirical_measures(experiment, regular, onOtherDataSet=onOtherDataSet, vivid_track_offset=vivid_track_offset) for experiment in experiments}).T
        empirical_measures['mean interval'] = empirical_measures['mean interval'] * minutes_per_timepoint
        empirical_measures['min interval'] = pd.Series({experiment: experiment_manager.theoretical_parameters[experiment]['min'] for experiment in experiments})
        empirical_measures['minutes per timepoint'] = minutes_per_timepoint

        empirical_measures['input entropy [bit/h]'] = empirical_measures['input entropy'] * 60 / minutes_per_timepoint 
        empirical_measures['input entropy assuming independent [bit/h]'] = empirical_measures['input entropy assuming independent'] * 60 / minutes_per_timepoint 
        empirical_measures['input entropy correction [bit/h]'] = empirical_measures['input entropy assuming independent [bit/h]'] - empirical_measures['input entropy [bit/h]']

        empirical_measures['transmitted information [bit/h]'] = mutual_information - empirical_measures['input entropy correction [bit/h]']

        if not regular:
            return empirical_measures['transmitted information [bit/h]'].agg(['mean', 'std', 'size'])
        else:
            return empirical_measures.groupby('minutes per timepoint')['transmitted information [bit/h]'].mean().agg(['mean', 'std', 'size'])

    
    mi_per_vivid_track = pd.DataFrame({vivid_track_offset: get_average_mi_for_group(experiments, regular, onOtherDataSet, vivid_track_offset) for vivid_track_offset in np.linspace(0, .5, 6)}).T
        
     
    plt.figure('figS3A', figsize=(5+1,3.5))
    color_list = [ 'maroon', 'purple',  'slateblue']
    symbol_list = ['^', 's', 'o']
    
    plt.errorbar(
        x=mi_per_vivid_track.index,
        y=mi_per_vivid_track['mean'],
        yerr=mi_per_vivid_track['std']/np.sqrt(mi_per_vivid_track['size' ]),
        marker=symbol_list[group_it], ls='--', color=color_list[group_it], label=title, capsize=3, ecolor='k', barsabove=False
        )

    bottom_of_the_best, top_of_the_best = (lambda x: (min(x, bottom_of_the_best), max(x, top_of_the_best)))(mi_per_vivid_track['mean'][0.2])

# remove errorbars from legend
handles, labels = plt.gca().get_legend_handles_labels()
handles = [h[0] for h in handles]
plt.gca().legend(handles, labels, loc='lower right', fontsize='large', numpoints=1)
    
plt.ylabel('Information transmission rate [b/h]', fontsize='large')
plt.xlabel('Percentage of rejected cells $f$', fontsize='large')
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0%}"))
plt.yticks(fontsize='large')
plt.xticks(fontsize='large')
plt.ylim(0,8)

left_of_the_best, right_of_the_best = [.2, .2]
plt.gca().add_patch(patches.Rectangle((left_of_the_best-(0.02), bottom_of_the_best-0.5), right_of_the_best-left_of_the_best+(0.04), top_of_the_best-bottom_of_the_best+1, facecolor=(.6,.6,.6), alpha=0.2, edgecolor='k'))


PART_OF_FIG3 = False
if PART_OF_FIG3:
    plt.subplots_adjust(bottom=0.2, left=0.02, right=0.7)
    plt.annotate('E', xy=(0.0,.98), xycoords='figure fraction', fontsize='xx-large', verticalalignment='top')
else:
    plt.subplots_adjust(bottom=0.2)
    plt.annotate('', xy=(-0.1,.98), xycoords='figure fraction', fontsize='xx-large', verticalalignment='top')



plt.savefig(output_path / 'figS3A.svg')
    
plt.show()

        

    

