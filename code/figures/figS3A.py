import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

from matplotlib import cm
from matplotlib.ticker import FuncFormatter


sys.path.append(str(Path(__file__).parent.parent))

from core import experiment_manager, factory
from core.steps import computing_information_transmission, get_timeline, computing_information_transmission_full_3bit_matrix

from figures.local_config import figure_output_path
from figures.figure_settings import *
from integrity import check_and_fetch
from utils.math_utils import input_entropy

check_and_fetch.check_and_fetch_necessary()

output_path = Path(figure_output_path).absolute() / "figS3/automatic/"
output_path.mkdir(parents=True, exist_ok=True)

tab10cmap = cm.get_cmap('tab10')

t_pos = 4
onGoodTracks = False
onOtherDataSet = False 
regular = False
remove_first_pulses = 0
simulate = False
yesno = False
plot_errorbars = False
with_voting = True

with_neighbors = True


positions_from = 2
positions_to = 9


bottom_of_the_best = +np.inf
top_of_the_best = -np.inf

for group_it,(title, figname, settings, experiments, additional_parameters) in enumerate((
    
    
    ('Interval encoding with a minimal gap', 'fig4B.svg', {'regular': False, 'onOtherDataSet': False}, 
           [experiment for experiment in experiment_manager.chosen_experiments_interval_with_gap if experiment in experiment_manager.best_experiments],
    {},
    ),
    
    ('Interval encoding', 'fig3B.svg', {'regular': False, 'onOtherDataSet': False},
        [experiment for experiment in experiment_manager.chosen_experiments_interval if experiment in experiment_manager.best_experiments],

    {},
    ),
    ('Binary encoding', 'fig2B.svg', {'regular': True, 'onOtherDataSet': False, 'remove_first_pulses': 0},
        [experiment for experiment in experiment_manager.chosen_experiments_pseudorandom if experiment in experiment_manager.best_experiments],
    {},
    ),

)): 

    if 'regular' in settings.keys():
        regular = settings['regular']
    if 'onOtherDataSet' in settings.keys():
        onOtherDataSet = settings['onOtherDataSet']
    if 'removeFirstPulses' in settings.keys():
        removeFirstPulses = settings['removeFirstPulses']

    
    outputs = {}
    outputs_aux = {}


    for vivid_track_offset in np.linspace(0, .5, 6):
        
        
        for it_experiment, experiment in enumerate(experiments):

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
                'target_position': t_pos,
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
                    'timeline_extraction_method': 'normal',
                    'loss_source_determination_method': 'sequential_averaged',

                } if regular else {}),
                
                'vivid_track_criteria': [
                    ('', 'index', 'lt', 500),
                    ('std_dQ2', 'rank', 'gt', vivid_track_offset),
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
                # factory.do_the_analysis(parameters, parameters1, regular, onOtherDataSet, onGoodTracks, yesno).step(get_binary_results).step(interpretation_new).step(computing_information_transmission_with_neighbors).add_output_file('entropy', '.xlsx')
                factory.do_the_analysis(parameters, parameters1,
                        regular, onOtherDataSet, onGoodTracks, yesno
                    ).step(get_timeline).step(computing_information_transmission_full_3bit_matrix)
                
                if not regular else
                factory.do_the_analysis(parameters, parameters1,
                        regular, onOtherDataSet, onGoodTracks, yesno
                    ).step(get_timeline).step(computing_information_transmission)
            )

            blinks : pd.Series =  chain.load_file('blinks')
            previous_pulse_lengths :pd.Series = chain.load_file('previous_pulse_lengths')
            results = chain.load_file('prediction_results')
            this_pulse_length = previous_pulse_lengths.shift(-1).rename('this_pulse_length')
            next_pulse_length = previous_pulse_lengths.shift(-2).rename('next_pulse_length')
            track_information = chain.load_file('track_information').join(chain.load_file('track_info'))
            overall_empirical = chain.load_file('information_overall_empirical')
            overall_theoretical = chain.load_file('information_overall_theoretical')
            per_track_empirical = chain.load_file('information_per_track_empirical')
            per_track_theoretical = chain.load_file('information_per_track_theoretical')


            
            
            
            output = pd.Series({
                # 'experiment': experiment,
                'mean': theoretical_mean,
                'min': theoretical_min,
                'exp_mean': theoretical_exp_mean,
                **parameters,
                'minutes_per_timepoint': minutes_per_timepoint,
                'THEORETICAL PARAMETERS': '',
                'theoretical input period [timepoints]': theoretical_mean,
                'theoretical input period [minutes]': theoretical_mean*minutes_per_timepoint,
                'theoretical input entropy [b/timepoint]': theoretical_input_entropy,
                'theoretical input entropy assuming Poisson process [b/timepoint]': theoretical_input_entropy_assuming_poisson,
                'ACTUAL PARAMETERS': '',
                'empirical input period [timepoints]': 1/overall_empirical['input_pulses'],
                'empirical input period [minutes]': minutes_per_timepoint/overall_empirical['input_pulses'],
                'empirical input entropy [b/timepoint]': input_entropy(theoretical_min, 1/overall_empirical['input_pulses'] - theoretical_min),
                'empirical input entropy assuming Poisson process [b/timepoint]': input_entropy(0, 1/overall_empirical['input_pulses']),
                'empirical/theoretical input freqency': overall_empirical['input_pulses'] / theoretical_input_frequency,
                'tracks taken': '',#len(results.index.get_level_values('track_id').unique()),
                'EMPIRICAL RESULTS': '',
                '(compute confusion matrix (CM) \n -> average among cells \n-> guess input frequency \n -> compute channel capacity)': '',
                **overall_empirical.rename(lambda x: str(x) + '_emp').to_dict(),
                'RESULTS RESCALED FOR THEORETICAL PULSE FREQUENCY': '',
                '(compute confusion matrix (CM) \n -> average among cells \n -> rescale CM to get input frequency as in theory \n -> compute channel capacity)': '',
                **overall_theoretical.rename(lambda x: str(x) + '_rescaled').to_dict(),
                'AVERAGE PER-TRACK PARAMETERS (EMPIRICAL)': '',
                '(compute confusion matrix (CM) \n -> guess input frequency \n -> compute channel capacity \n -> average among cells)': '',
                **per_track_empirical.drop(['channel_capacity[min/b]', 'channel_capacity_assuming_poisson[min/b]'], axis=1).mean().append(pd.Series({            
                    'channel_capacity[min/b]' : minutes_per_timepoint/per_track_empirical['channel_capacity[b/timepoint]'].mean(),
                    'channel_capacity_assuming_poisson[min/b]' : minutes_per_timepoint/per_track_empirical['channel_capacity_assuming_poisson[b/timepoint]'].mean(),
                })).rename(lambda x: str(x) + '_avg_per_track_emp').to_dict(),
                'AVERAGE PER-TRACK PARAMETERS (RESCALED)': '',
                '(compute confusion matrix (CM) \n-> rescale CM to get input frequency as in theory \n -> compute channel capacity \n -> average among cells)': '',
                **per_track_theoretical.drop(['channel_capacity[min/b]', 'channel_capacity_assuming_poisson[min/b]'], axis=1).mean().append(pd.Series({            
                    'channel_capacity[min/b]' : minutes_per_timepoint/per_track_theoretical['channel_capacity[b/timepoint]'].mean(),
                    'channel_capacity_assuming_poisson[min/b]' : minutes_per_timepoint/per_track_theoretical['channel_capacity_assuming_poisson[b/timepoint]'].mean(),
                })).rename(lambda x: str(x) + '_avg_per_track_rescaled').to_dict(),
                # 'mean_track_length': scores_with_correction_for_consecutive.groupby('track_id')['length'].mean().mean()
            }, name=experiment)
            print(output)

            outputs = {**outputs, (experiment, vivid_track_offset): output}
            

            # print(per_track_empirical)
                
            outputs_aux = {**outputs_aux, 
                (experiment, vivid_track_offset): pd.Series({
                'mean': theoretical_mean,
                'min': theoretical_min,
                'exp_mean': theoretical_exp_mean,
                **parameters,
                'minutes_per_timepoint': minutes_per_timepoint,
                'AVERAGE PER-TRACK PARAMETERS (EMPIRICAL)': '',
                '(compute confusion matrix (CM) \n -> guess input frequency \n -> compute channel capacity \n -> average among cells)': '',
                **per_track_empirical[(lambda x: x > 1/2*x.mean())(per_track_empirical.join(track_information, on='track_id')['std_dQ2'])].drop(['channel_capacity[min/b]', 'channel_capacity_assuming_poisson[min/b]'], axis=1).mean().append(pd.Series({            
                    'channel_capacity[min/b]' : minutes_per_timepoint/per_track_empirical['channel_capacity[b/timepoint]'].mean(),
                    'channel_capacity_assuming_poisson[min/b]' : minutes_per_timepoint/per_track_empirical['channel_capacity_assuming_poisson[b/timepoint]'].mean(),
                })).rename(lambda x: str(x) + '_avg_per_track_emp').to_dict(),
                'AVERAGE PER-TRACK PARAMETERS (RESCALED)': '',
                '(compute confusion matrix (CM) \n-> rescale CM to get input frequency as in theory \n -> compute channel capacity \n -> average among cells)': '',
                **per_track_theoretical[(lambda x: x > 1/2*x.mean())(per_track_theoretical.join(track_information, on='track_id')['std_dQ2'])].drop(['channel_capacity[min/b]', 'channel_capacity_assuming_poisson[min/b]'], axis=1).mean().append(pd.Series({            
                    'channel_capacity[min/b]' : minutes_per_timepoint/per_track_theoretical['channel_capacity[b/timepoint]'].mean(),
                    'channel_capacity_assuming_poisson[min/b]' : minutes_per_timepoint/per_track_theoretical['channel_capacity_assuming_poisson[b/timepoint]'].mean(),
                })).rename(lambda x: str(x) + '_avg_per_track_rescaled').to_dict(),
                    # 'mean_track_length': scores_with_correction_for_consecutive.groupby('track_id')['length'].mean().mean()
                }, name=experiment)
            }


        
    outputs_df = pd.DataFrame(outputs).T.rename_axis(index=['experiment', 'vivid_track_offset'])
    outputs_aux_df = pd.DataFrame(outputs_aux).T.rename_axis(index=['experiment', 'vivid_track_offset'])
    if regular:
        outputs_df['information lost due to inacurate detections[b/timepoint]_emp'] = 0
        outputs_df['information lost due to inacurate detections[b/timepoint]_avg_per_track_emp'] = 0
        outputs_aux_df['information lost due to inacurate detections[b/timepoint]_avg_per_track_emp'] = 0
    
    print(outputs_df)
    # for _,x in outputs_df.groupby(['empirical input period [minutes]']):
    #     print(x)
    #     print(x[['empirical input period [minutes]', 'minutes_per_timepoint', 'channel_capacity[b/timepoint]_emp',  'information lost due to inacurate detections[b/timepoint]_emp', 'information lost due to false detections[b/timepoint]_emp', 'information lost due to missed pulses[b/timepoint]_emp']])
    if regular: 
        selected_output_fields = outputs_df[['minutes_per_timepoint', 'empirical input period [minutes]', 'min', 'channel_capacity[b/timepoint]_emp',  'information lost due to inacurate detections[b/timepoint]_emp', 'information lost due to false detections[b/timepoint]_emp', 'information lost due to missed pulses[b/timepoint]_emp', 'channel_capacity[b/h]_emp', 'TP_emp', 'FP_emp', 'FN_emp']].astype('float').groupby(['minutes_per_timepoint', 'vivid_track_offset']).mean().reset_index().sort_values(['minutes_per_timepoint', 'vivid_track_offset'], ascending=True).rename(columns={'minutes_per_timepoint': 'experiment'}).set_index(['experiment', 'vivid_track_offset'])
    else:
        selected_output_fields = outputs_df.reset_index().sort_values(['experiment', 'vivid_track_offset'], ascending=True).set_index(['experiment', 'vivid_track_offset'])




    plt.figure('fig S1E', figsize=(5+1,3.5))
    # plt.subplot(1, 3, group_it+1)
    print(selected_output_fields)
    print(selected_output_fields.columns)
    color_list = [ 'maroon', 'purple',  'slateblue']
    symbol_list = ['^', 's', 'o']
    # outputs_df2.reset_index().groupby('experiment').plot('vivid_track_offset', 'channel_capacity[b/h]_emp', marker='o', ax=plt.gca())
    
    output_per_vivid_track_offset = selected_output_fields['channel_capacity[b/h]_emp'].astype('float').groupby('vivid_track_offset').agg(['count', 'mean', 'std'])
    output_per_vivid_track_offset['mean'].plot(marker=symbol_list[group_it], ax=plt.gca(), ls='--', color=color_list[group_it], label=title)
    plt.errorbar(output_per_vivid_track_offset.index, output_per_vivid_track_offset['mean'], yerr=output_per_vivid_track_offset['std']/np.sqrt(output_per_vivid_track_offset['count']), fmt='none', color=color_list[group_it], label=None,  capsize=3, ecolor='k')
    print(output_per_vivid_track_offset)
    bottom_of_the_best, top_of_the_best = (lambda x: (min(x, bottom_of_the_best), max(x, top_of_the_best)))(output_per_vivid_track_offset['mean'][0.2])


plt.legend(loc='lower right', fontsize='large')

# plt.legend(outputs_df2.index.get_level_values('experiment').unique(), title='Period' if regular else 'experiment')
plt.ylabel('Information transmission rate [b/h]', fontsize='large')
plt.xlabel('Percentage of rejected cells $f$', fontsize='large')
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0%}"))
plt.yticks(fontsize='large')
plt.xticks(fontsize='large')
plt.ylim(0,8)

left_of_the_best, right_of_the_best = [.2, .2]
plt.gca().add_patch(plt.Rectangle((left_of_the_best-(0.02), bottom_of_the_best-0.5), right_of_the_best-left_of_the_best+(0.04), top_of_the_best-bottom_of_the_best+1, facecolor=(.6,.6,.6), alpha=0.2, edgecolor='k'))


PART_OF_FIG3 = False
if PART_OF_FIG3:
    plt.subplots_adjust(bottom=0.2, left=0.02, right=0.7)
    plt.annotate('E', xy=(0.0,.98), xycoords='figure fraction', fontsize='xx-large', verticalalignment='top')
else:
    plt.subplots_adjust(bottom=0.2)
    plt.annotate('', xy=(-0.1,.98), xycoords='figure fraction', fontsize='xx-large', verticalalignment='top')



plt.savefig(output_path / 'figS3A.svg')
    
plt.show()

        

    

