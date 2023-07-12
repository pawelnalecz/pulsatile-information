import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core import experiment_manager, factory
from core.steps import MI_from_CM_grouped

from figures.local_config import figure_output_path
from figures.figure_settings import *
from integrity import check_and_fetch

check_and_fetch.check_and_fetch_necessary()

output_path = Path(figure_output_path).absolute() / "figS3"
output_path.mkdir(parents=True, exist_ok=True)

learning = True


for group_it,(title, figletter, xaxis, regular, onOtherDataSet, experiments, additional_parameters) in enumerate((
    
    ('Binary encoding', 'B', 'minutes_per_timepoint', True, True,
        experiment_manager.chosen_experiments_pseudorandom,
    {},
    ),
    ('Interval encoding', 'C',  'empirical input period [minutes]' , False, False,
        experiment_manager.chosen_experiments_interval,
    {},
    ),
    ('Interval encoding with a minimal gap', 'D', 'min', False, False,
        experiment_manager.chosen_experiments_interval_with_gap,
    {},
    ),

)):
    
    def get_parameters(experiment, regular, **kwargs):
        return {
            **experiment_manager.default_parameters,
            **experiment_manager.experiments[experiment],
            'theoretical_parameters': experiment_manager.theoretical_parameters[experiment],
            'trim_end': experiment_manager.trim_end(experiment),
            'fields_reduced_with_confusion_matrix': ['time_point'],
            'fields_for_mi_computation_grouping': ['track_id'],
            'correct_consecutive': 2,

            **({
                'correct_consecutive': 0,
                'n_pulses': 19,
                'pulse_length': experiment_manager.theoretical_parameters[experiment]['minutes_per_timepoint'],
                'train_on_other_experiment': onOtherDataSet,
                'r_slice_length': 1,

            } if regular else {}),
            
            'vivid_track_criteria': [
                ('', 'index', 'lt', 500),
            ],

            **kwargs,
        }
    

    additional_preselection = [
        ('', 'index', 'lt', 500),
        ('std_dQ2', 'rank', 'gt', 0.2),
    ]

    chains = {
        experiment: factory.compute_information_transmission(regular, learning)(
            parameters=get_parameters(experiment, regular), 
            parameters1=get_parameters(experiment_manager.get_complementary_experiment(experiment), regular) if onOtherDataSet else None
            ).step(MI_from_CM_grouped) for experiment in experiments
    }

    chains_with_preselection = {
        experiment: factory.compute_information_transmission(regular, learning)(
            parameters=get_parameters(experiment, regular, vivid_track_criteria=additional_preselection), 
            parameters1=get_parameters(experiment_manager.get_complementary_experiment(experiment), regular, vivid_track_criteria=additional_preselection) if onOtherDataSet else None
            ).step(MI_from_CM_grouped) for experiment in experiments
    }


    minutes_per_timepoint = pd.Series({experiment: experiment_manager.theoretical_parameters[experiment]['minutes_per_timepoint'] for experiment in experiments}, name='minutes per timepoint')
    minutes_per_timepoint.index.name = 'experiment'

    empirical_measures = pd.DataFrame({experiment: chain.load_file('empirical_measures') for experiment,chain in chains.items()}).T
    empirical_measures.index.name = 'experiment'
    empirical_measures['mean interval'] = empirical_measures['mean interval'] * minutes_per_timepoint
    empirical_measures['min interval'] = pd.Series({experiment: experiment_manager.theoretical_parameters[experiment]['min'] for experiment in experiments})
    empirical_measures['minutes per timepoint'] = minutes_per_timepoint

    empirical_measures['input entropy [bit/h]'] = empirical_measures['input entropy'] * 60 / minutes_per_timepoint 
    empirical_measures['input entropy assuming independent [bit/h]'] = empirical_measures['input entropy assuming independent'] * 60 / minutes_per_timepoint 
    empirical_measures['input entropy correction [bit/h]'] = empirical_measures['input entropy assuming independent [bit/h]'] - empirical_measures['input entropy [bit/h]']

    mutual_information_per_track = pd.concat((
            chain.load_file('mutual_informations_grouped') for chain in chains.values()),
            names=['experiment'],
            keys=experiments,
        ) * 60 / minutes_per_timepoint - empirical_measures['input entropy correction [bit/h]']
    mutual_information_per_track.name = 'transmitted information [bit/h]'
    
    mutual_information_per_track_with_preselection = pd.concat((
        chain.load_file('mutual_informations_grouped') for chain in chains_with_preselection.values()),
        names=['experiment'],
        keys=experiments,
    ) * 60 / minutes_per_timepoint - empirical_measures['input entropy correction [bit/h]']
    mutual_information_per_track_with_preselection.name = 'transmitted information [bit/h]'

    print(mutual_information_per_track)

    if regular:
        mutual_information_per_track = mutual_information_per_track.to_frame().join(minutes_per_timepoint, on='experiment')#.set_index('minutes per timepoint', append=True)['transmitted information [bit/h]']
        mutual_information_per_track_with_preselection = mutual_information_per_track_with_preselection.to_frame().join(minutes_per_timepoint, on='experiment')#.set_index('minutes per timepoint', append=True)['transmitted information [bit/h]']
    else:
        mutual_information_per_track = mutual_information_per_track.to_frame().join(empirical_measures, on='experiment')#.set_index('mean interval', append=True).sort_index(level=['mean interval', 'track_id']).reset_index('mean interval')['transmitted information [bit/h]']
        mutual_information_per_track_with_preselection = mutual_information_per_track_with_preselection.to_frame().join(empirical_measures, on='experiment')#.set_index('mean interval', append=True).sort_index(level=['mean interval', 'track_id']).reset_index('mean interval')['transmitted information [bit/h]']


    used_n_of_charts = len(mutual_information_per_track.index.get_level_values('experiment').unique() if not regular else mutual_information_per_track['minutes per timepoint'].unique())
    n_of_charts = used_n_of_charts#len(experiment_manager.chosen_experiments_interval_with_gap)
    fig, ax = plt.subplots(1, used_n_of_charts, num=f'Fig S3{figletter} -- Histograms -- ' + title, figsize=(5,1+0.5*n_of_charts), sharex=True, sharey=False)
    for it_experiment, ((label, mi_no_preselection), (_, mi_with_preselection))  in enumerate(zip(
            mutual_information_per_track.groupby(['mean interval', 'min interval'] if not regular else 'minutes per timepoint')['transmitted information [bit/h]'], 
            mutual_information_per_track_with_preselection.groupby(['mean interval', 'min interval'] if not regular else 'minutes per timepoint')['transmitted information [bit/h]'], 
            )):
        print(label)
        plt.subplot(used_n_of_charts, 1, it_experiment+1)
        mi_no_preselection.plot.hist(bins=np.linspace(-5-0.1, 20-0.1, 26), color='orange')
        mi_with_preselection.plot.hist(bins=np.linspace(-5-0.1, 20-0.1, 26), color='slateblue')

        plt.ylabel(
            f"{label[0]:.0f} [{label[1]:.0f}] min" if not regular else f"{label}",
            rotation=0,
            horizontalalignment='left',
            verticalalignment='center',
            fontdict=dict(fontweight='bold'),
            labelpad=74 if not regular else 25,
            fontsize='large')
        plt.xlabel('Information transmission rate [bit/h]', fontsize='large')
        plt.yticks([])
        plt.xticks(fontsize='large')
        plt.subplots_adjust(left=0.3, top=(used_n_of_charts+1)/(n_of_charts+2), bottom=1/(n_of_charts+2))#.23
        # plt.annotate(f"{mi_with_preselection.mean()}", (0,plt.ylim()[1]/2))
    fig.add_subplot(111, frameon=False)
    plt.title(title, fontsize='x-large')
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel('Mean interval betwean pulses $\\tau_{geom}+\\tau_{gap}$ [min]\n minimal gap $\\tau_{gap}$ [min] ' if not regular else 'Clock period $\\tau_{clock}$ [min]', labelpad=60 if not regular else 20, fontsize='large') 
    plt.annotate(figletter, xy=(0.05 if regular else -0.02, .98 if group_it != 1 else 1.07), xycoords='figure fraction', fontsize='xx-large', verticalalignment='top', fontweight='bold')
    plt.savefig(output_path / f"FigS3{figletter}.svg")


plt.show()
