import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from matplotlib import cm

sys.path.append(str(Path(__file__).parent.parent))

from core import experiment_manager, factory
from core.steps import find_loss_sources_based_on_timeline

from figures.local_config import figure_output_path
from figures.figure_settings import *
from integrity import check_and_fetch

check_and_fetch.check_and_fetch_necessary()

output_path = Path(figure_output_path).absolute() / "fig2"
output_path.mkdir(parents=True, exist_ok=True)


fontsize = 'medium'
learning = True


xaxis_to_label = {
    'minutes_per_timepoint': 'Period [min]', 
    'empirical input period [minutes]': 'Mean interval between pulses [min]', 
    'min': 'Minimal gap [min]'
    }

xaxis_to_xlim = {
    'minutes_per_timepoint': (0,35), 
    'empirical input period [minutes]': (0,70), 
    'min': (0,40)
    }

for group_it,(title, figname, figletter, figletter_overall_figure, xaxis, regular, onOtherDataSet, experiments, additional_parameters) in enumerate((
    
    ('Binary encoding', 'fig2B.svg', 'B', 'A', 'minutes_per_timepoint', True, True, # onOtherDataSet should be True accoring to Methods
        experiment_manager.chosen_experiments_pseudorandom,
    {},
    ),
    ('Interval encoding', 'fig2D.svg', 'D', 'B',  'empirical input period [minutes]', False, False,
        experiment_manager.chosen_experiments_interval,
    {},
    ),
    ('Interval encoding with minimal gap', 'fig2F.svg', 'F', 'C', 'empirical input period [minutes]', False, False, #min
        experiment_manager.chosen_experiments_interval_with_gap,
    {},
    ),

)):
    figname_without_extension = figname[:-4]

    def get_parameters(experiment, regular, onOtherDataSet):
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


    def get_chain(experiment, regular, learning, onOtherDataSet):
        parameters = get_parameters(experiment, regular, onOtherDataSet)
        parameters1 = get_complementary_parameters(parameters, experiment) if parameters['train_on_other_experiment'] else None
        chain = factory.compute_information_transmission(regular=regular, learning=learning)(parameters=parameters, parameters1=parameters1) \
            .step(find_loss_sources_based_on_timeline)
        return chain


    def get_mi(experiment, regular, learning, onOtherDataSet, return_iterations=False):
        parameters = get_parameters(experiment, regular, onOtherDataSet)
        minutes_per_timepoint = parameters['theoretical_parameters']['minutes_per_timepoint']
        chain = get_chain(experiment, regular, learning, onOtherDataSet)
        if return_iterations:
            return (
                chain.load_file('mutual_information') / parameters['s_slice_length'] * 60 / minutes_per_timepoint, 
                [mi / parameters['s_slice_length'] * 60 / minutes_per_timepoint for mi in chain.load_file('mutual_informations')]
                )
        return chain.load_file('mutual_information') / parameters['s_slice_length'] * 60 / minutes_per_timepoint


    def get_loss_sources(experiment, regular, learning, onOtherDataSet):
        parameters = get_parameters(experiment, regular, onOtherDataSet)
        minutes_per_timepoint = parameters['theoretical_parameters']['minutes_per_timepoint']
        chain = get_chain(experiment, regular, learning, onOtherDataSet)
        return chain.load_file('loss_sources') / parameters['s_slice_length'] * 60 / minutes_per_timepoint


    def get_empirical_measures(experiment, regular, learning, onOtherDataSet):
        chain = get_chain(experiment, regular, learning, onOtherDataSet)
        return chain.load_file('empirical_measures')


    minutes_per_timepoint = pd.Series({experiment: experiment_manager.theoretical_parameters[experiment]['minutes_per_timepoint'] for experiment in experiments}, name='minutes_per_timepoint')
    
    mutual_information_s = {experiment: get_mi(experiment, regular=regular, learning=learning, onOtherDataSet=onOtherDataSet, return_iterations=True) for experiment in experiments}
    mutual_information = pd.Series({experiment: mutual_information_s[experiment][0] for experiment in experiments}, name='MI [bit/h]')
    mutual_information_std = pd.Series({experiment: np.std(mutual_information_s[experiment][1]) for experiment in experiments})


    empirical_measures = pd.DataFrame({experiment: get_empirical_measures(experiment, regular, learning, onOtherDataSet) for experiment in experiments}).T
    empirical_measures['mean interval'] = empirical_measures['mean interval'] * minutes_per_timepoint
    empirical_measures['min interval'] = pd.Series({experiment: experiment_manager.theoretical_parameters[experiment]['min'] for experiment in experiments})
    empirical_measures['minutes per timepoint'] = minutes_per_timepoint

    empirical_measures['input entropy [bit/h]'] = empirical_measures['input entropy'] * 60 / minutes_per_timepoint 
    empirical_measures['input entropy assuming independent [bit/h]'] = empirical_measures['input entropy assuming independent'] * 60 / minutes_per_timepoint 
    empirical_measures['input entropy correction [bit/h]'] = empirical_measures['input entropy assuming independent [bit/h]'] - empirical_measures['input entropy [bit/h]']

    empirical_measures['transmitted information [bit/h]'] = mutual_information - empirical_measures['input entropy correction [bit/h]']
    
    loss_sources = pd.DataFrame({experiment: get_loss_sources(experiment, regular, learning, onOtherDataSet) for experiment in experiments}).T

    if 'ID' not in loss_sources.columns:
        loss_sources['ID'] = 0


    empirical_measures = pd.concat([empirical_measures, loss_sources], axis='columns')
    
    interesting_measures = empirical_measures[[
        'min interval',
        'mean interval',
        'minutes per timepoint',
        'transmitted information [bit/h]',
        'input entropy [bit/h]',
        'ID',
        'FN',
        'FP',
        ]]


    if regular:
        interesting_measures = interesting_measures.groupby('minutes per timepoint').mean().sort_index().reset_index()
    else:
        interesting_measures = interesting_measures.sort_values('mean interval')


    plt.figure(figname)
    interesting_measures.to_excel(output_path / (figname_without_extension + '.xlsx'))
    interesting_measures.to_html(output_path / (figname_without_extension + '.html'), float_format='{:.2f}'.format)
    interesting_measures[[
        'transmitted information [bit/h]',
        'ID',
        'FN',
        'FP',
    ]].plot.bar(stacked=True, figsize=(4.5,3.5), color=['k', (0., 178/256, 178/256), (87/256, 123/256, 255/256), (189/256, 0., 189/256)], ax=plt.gca())#color=cm.get_cmap('tab10')([0, 1, 3, 2]))#['blue', 'lightskyblue', 'olivedrab', 'lightsalmon']
    plt.ylim(0,26)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('Information transmission rate [bit/h]', fontsize=fontsize)


    if title == 'Binary encoding':
        plt.xticks(
            plt.xticks()[0],
            [f"{minutes_per_timepoint:.3g}" for minutes_per_timepoint in interesting_measures['minutes per timepoint']],
            rotation=0,  fontsize=fontsize)
        plt.xlabel('Clock period [min]', fontsize=fontsize)
    elif title == 'Interval encoding':
        plt.xticks(
            plt.xticks()[0],
            [f"{row['mean interval']:.0f}" for experiment,row in interesting_measures.iterrows()],
            rotation=0, fontsize=fontsize)
        plt.xlabel('Mean interval between pulses [min]', fontsize=fontsize)
    elif title == 'Interval encoding with minimal gap':
        plt.xticks(
            plt.xticks()[0],
            [f"{row['mean interval'] :.0f} \n[{row['min interval']:.0f}]" for experiment,row in interesting_measures.iterrows()],
            rotation=0, fontsize=fontsize)
        plt.xlabel('Mean interval between pulses [min]\n[minimal gap $T_{gap}$] [min]', fontsize=fontsize)

    if not regular:
        plt.legend(reversed(plt.gca().get_legend_handles_labels()[0]), reversed(['transmitted', 'lost due to inacurate detection', 'lost due to missed pulses',  'lost due to false detections']), fontsize='medium')
    else:
        plt.legend(reversed([handle for handle,label in zip(*plt.gca().get_legend_handles_labels()) if label != 'ID']), reversed(['transmitted', 'lost due to missed pulses',  'lost due to false detections']), fontsize='medium')

    plt.subplots_adjust(left=0.15, bottom=0.25)
    plt.annotate(figletter, xy=(0.02,.98), xycoords='figure fraction', fontsize='xx-large', verticalalignment='top', fontweight='bold')
    
    plt.savefig(output_path / figname)

    print(interesting_measures)

plt.show()

        

    

