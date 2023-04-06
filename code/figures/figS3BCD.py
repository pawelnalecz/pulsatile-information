import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

sys.path.append(str(Path(__file__).parent.parent))

from core import experiment_manager, factory
from core.steps import computing_information_transmission_full_3bit_matrix, computing_information_transmission, get_timeline

from figures.local_config import figure_output_path
from figures.figure_settings import *
from integrity import check_and_fetch
from utils.math_utils import input_entropy

check_and_fetch.check_and_fetch_necessary()

output_path = Path(figure_output_path).absolute() / "figS3/automatic/"
output_path.mkdir(parents=True, exist_ok=True)


onGoodTracks = False
yesno = False
with_voting = True



for group_it,(title, figletter, xaxis, settings, experiments, additional_parameters) in enumerate((
    
    ('Binary encoding', 'B', 'minutes_per_timepoint', {'regular': True, 'onOtherDataSet': False, 'yesno': True, 'remove_first_pulses': 0},
        experiment_manager.chosen_experiments_pseudorandom,
    {},
    ),
    ('Interval encoding', 'C',  'empirical input period [minutes]' , {'regular': False, 'onOtherDataSet': False, 'yesno': False,},
        experiment_manager.chosen_experiments_interval,
    {},
    ),
    ('Interval encoding with a minimal gap', 'D', 'min', {'regular': False, 'onOtherDataSet': False, 'yesno': False},
        experiment_manager.chosen_experiments_interval_with_gap,
    {},
    ),

)): 

    if 'regular' in settings.keys():
        regular = settings['regular']
    if 'onOtherDataSet' in settings.keys():
        onOtherDataSet = settings['onOtherDataSet']
    if 'removeFirstPulses' in settings.keys():
        removeFirstPulses = settings['removeFirstPulses']
    if 'yesno' in settings.keys():
        yesno = settings['yesno']

    outputs = {}
    outputs_aux = {}

    
    for it_experiment, experiment in enumerate(experiments):

        print('.............. STARTING ' + experiment + '................') 


        theoretical_min = experiment_manager.theoretical_parameters[experiment]['min']
        theoretical_exp_mean = experiment_manager.theoretical_parameters[experiment]['exp_mean']
        theoretical_mean = theoretical_min + theoretical_exp_mean
        minutes_per_timepoint = experiment_manager.theoretical_parameters[experiment]['minutes_per_timepoint']
        
        theoretical_input_frequency : float= 1/theoretical_mean
        theoretical_input_entropy : float = input_entropy(theoretical_min, theoretical_exp_mean)
        theoretical_input_entropy_assuming_poisson : float = input_entropy(0, 1/theoretical_input_frequency)


        parameters = {
            **experiment_manager.default_parameters,
            **experiment_manager.experiments[experiment],
            'theoretical_parameters': experiment_manager.theoretical_parameters[experiment],
            'train_on': 'other_tracks' if not onOtherDataSet else 'same',
            'trim_end': (-1, int(np.floor(experiment_manager.theoretical_parameters[experiment]['min'] + experiment_manager.theoretical_parameters[experiment]['exp_mean']))),
            'timeline_extraction_method' : 'normal' if (not with_voting or yesno ) else 'subsequent',
    


            **({
                'remove_first_pulses': 0,
                'remove_break': 0,
                'correct_consecutive': 0,
                'remove_shorter_than': 0,
                'yesno': True,
                'n_pulses': 19,
                'pulse_length': minutes_per_timepoint,

            } if regular else {}),
            
            'vivid_track_criteria': [
                ('', 'index', 'lt', 500),
            ],


            **additional_parameters,
        }

        if regular and onOtherDataSet:
            pos_text = re.search('pos[0-9]+_', experiment).group(0)
            pos = int(pos_text[3:-1])
            complementary_pos_text = f'pos{11-pos:02d}_'
        
        if onOtherDataSet:
            parameters1 = {**parameters, **experiment_manager.experiments['min20_optmeanb' if not regular else experiment.replace(pos_text, complementary_pos_text)] , 'good_track_offset': 0.85}
        else:
            parameters1=parameters

        chain = (
                (factory.do_the_analysis(parameters, parameters1, regular, onOtherDataSet, onGoodTracks, yesno).step(get_timeline)
                if not yesno or not with_voting else factory.get_voting_timeline(parameters, parameters1, regular, onOtherDataSet, onGoodTracks, yesno)
            ).step(computing_information_transmission_full_3bit_matrix)
            if not regular else
            #factory.do_the_analysis(parameters, parameters1, regular, onOtherDataSet, onGoodTracks, yesno).step(get_timeline).step(computing_information_transmission)
            factory.get_voting_timeline(parameters, parameters1, regular, onOtherDataSet, onGoodTracks, yesno).step(computing_information_transmission)
        )

        additional_preselection = [
                ('', 'index', 'lt', 500),
                ('std_dQ2', 'rank', 'gt', 0.2),
            ]


        chain2 = (
                (factory.do_the_analysis({**parameters, 'vivid_track_criteria': additional_preselection}, parameters1, regular, onOtherDataSet, onGoodTracks, yesno).step(get_timeline)
                if not yesno or not with_voting else factory.get_voting_timeline({**parameters, 'vivid_track_criteria': additional_preselection}, parameters1, regular, onOtherDataSet, onGoodTracks, yesno)
            ).step(computing_information_transmission_full_3bit_matrix)
            if not regular else
            #factory.do_the_analysis(parameters, parameters1, regular, onOtherDataSet, onGoodTracks, yesno).step(get_timeline).step(computing_information_transmission)
            factory.get_voting_timeline({**parameters, 'vivid_track_criteria': additional_preselection}, parameters1, regular, onOtherDataSet, onGoodTracks, yesno).step(computing_information_transmission)
        )



        outputs = {**outputs, experiment:
            { 
                **parameters, 
                **parameters['theoretical_parameters'], 
                'empirical input period [minutes]': minutes_per_timepoint/chain.load_file('information_overall_empirical')['input_pulses'],
                'per_track_empirical': chain.load_file('information_per_track_empirical'), 
                # 'per_track_empirical2': chain2.load_file('information_per_track_empirical'), 
                'vivid_tracks': chain2.load_file('vivid_tracks'),
            }}
    outputs_df = pd.DataFrame(outputs).T
    outputs_df.index.name = 'experiment'
    used_n_of_charts = len(outputs_df.groupby(['min', 'exp_mean', 'minutes_per_timepoint']).size()) if regular else len(outputs_df.groupby('experiment'))
    n_of_charts = used_n_of_charts#len(experiment_manager.chosen_experiments_interval_with_gap)
    # plt.figure('Fig S1F -- Histograms -- ' + title, figsize=(7,0.6*n_of_charts))
    fig, ax = plt.subplots(1, used_n_of_charts, num='Fig S1F -- Histograms -- ' + title, figsize=(5,1+0.5*n_of_charts), sharex=True, sharey=False)
    for it_experiment, (label, equivalent_experiments) in enumerate(outputs_df.sort_values(['minutes_per_timepoint', 'empirical input period [minutes]']).groupby(['minutes_per_timepoint'] if regular else ['experiment'], sort=False)):
        # print(equivalent_experiments[['minutes_per_timepoint', 'empirical input period [minutes]']])
        per_track_empirical = pd.concat([outputs[experiment]['per_track_empirical'] for experiment in equivalent_experiments.index])
        # per_track_empirical2 = pd.concat([outputs[experiment]['per_track_empirical2'] for experiment in equivalent_experiments.index])
        per_track_empirical_vivid_track = pd.concat([outputs[experiment]['per_track_empirical'].reindex(outputs[experiment]['vivid_tracks']) for experiment in equivalent_experiments.index])
        plt.subplot(used_n_of_charts, 1, it_experiment+1)
        per_track_empirical['channel_capacity[b/h]'].plot.hist(bins=np.linspace(-5-0.1, 20-0.1, 26), color='orange')
        per_track_empirical_vivid_track['channel_capacity[b/h]'].plot.hist(bins=np.linspace(-5-0.1, 20-0.1, 26), color='slateblue')
        # per_track_empirical2['channel_capacity[b/h]'].plot.hist(bins=np.linspace(-5-0.1, 20-0.1, 26), color='lightblue', histtype='step')

        plt.ylabel(f"{equivalent_experiments['empirical input period [minutes]'].mean():.0f} [{equivalent_experiments['min'].mean():.0f}] min"
            # f"${label[0]:.0f}$+Geom(${label[1]:.1f}$)"
            if not regular else f"{label}", rotation=0, horizontalalignment='left', verticalalignment='center', fontdict=dict(fontweight='bold'), labelpad=74 if not regular else 25, fontsize='large')
        # plt.ylim(0,300)
        plt.xlabel('Information transmission rate [bit/h]', fontsize='large')
        plt.yticks([])
        plt.xticks(fontsize='large')
        plt.subplots_adjust(left=0.3, top=(used_n_of_charts+1)/(n_of_charts+2), bottom=1/(n_of_charts+2))#.23
    fig.add_subplot(111, frameon=False)
    plt.title(title, fontsize='x-large')
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel('Mean interval betwean pulses $\\tau_{geom}+\\tau_{gap}$ [min]\n minimal gap $\\tau_{gap}$ [min] ' if not regular else 'Clock period $\\tau_{clock}$ [min]', labelpad=60 if not regular else 20, fontsize='large') 
    plt.annotate(figletter, xy=(0.05 if regular else -0.02, .98 if group_it != 1 else 1.07), xycoords='figure fraction', fontsize='xx-large', verticalalignment='top', fontweight='bold')
    plt.savefig(output_path / f"FigS3{figletter}.svg")


plt.show()
