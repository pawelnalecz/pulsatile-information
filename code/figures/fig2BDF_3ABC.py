import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

from matplotlib.patches import Ellipse
from matplotlib import cm

sys.path.append(str(Path(__file__).parent.parent))

from core import experiment_manager, factory
from core.steps import computing_information_transmission_full_3bit_matrix, computing_information_transmission, get_timeline

from figures.local_config import figure_output_path
from figures.figure_settings import *
from integrity import check_and_fetch
from utils.math_utils import input_entropy, optimal_tau

check_and_fetch.check_and_fetch_necessary()

output_path = Path(figure_output_path).absolute() / "fig234/automatic/"
output_path.mkdir(parents=True, exist_ok=True)



tab10cmap = cm.get_cmap('tab10')

t_pos = 4
onGoodTracks = False
onOtherDataSet = False 
regular = False
remove_first_pulses = 0
simulate = False
yesno = False
with_voting = True
PLOT_ERRORBARS = False

with_neighbors = True

CC_ylim_max = 8


positions_from = 2
positions_to = 9

fontsize = 'medium'
TICK_BEST_EXPERIMENTS_AVERAGE = True



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


for onPreselected, save_each, overall_figname in (
(True, True, 'fig3ABC.svg'),
):
    for group_it,(title, figname, figletter, figletter_overall_figure, xaxis, settings, experiments, additional_parameters) in enumerate((
        
        ('Binary encoding', 'fig2B.svg', 'B', 'A', 'minutes_per_timepoint', {'regular': True, 'yesno':True, 'onOtherDataSet': False, 'remove_first_pulses': 0},
            experiment_manager.chosen_experiments_pseudorandom,
        {},
        ),
        ('Interval encoding', 'fig2D.svg', 'D', 'B',  'empirical input period [minutes]' , {'regular': False, 'yesno':False, 'onOtherDataSet': False},
            experiment_manager.chosen_experiments_interval,
        {},
        ),
        ('Interval encoding with minimal gap', 'fig2F.svg', 'F', 'C', 'empirical input period [minutes]', {'regular': False, 'yesno':False,  'onOtherDataSet': False}, #min
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


        figname_without_extension = figname[:-4]
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
                    'loss_source_determination_method': 'sequential_averaged',


                } if regular else {}),
                
                'vivid_track_criteria': [
                    ('', 'index', 'lt', 500),
                    *([('std_dQ2', 'rank', 'gt', 0.2)] if onPreselected else []),
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
                factory.get_voting_timeline(parameters, parameters1, regular, onOtherDataSet, onGoodTracks, yesno).step(computing_information_transmission)
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

            print(chain.load_file('binary_timeline'))
            
            print(overall_empirical)

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
                'MEDIAN PER-TRACK PARAMETERS (EMPIRICAL)': '',
                '(compute confusion matrix (CM) \n -> guess input frequency \n -> compute channel capacity \n -> average among cells)': '',
                **per_track_empirical.drop(['channel_capacity[min/b]', 'channel_capacity_assuming_poisson[min/b]'], axis=1).median().append(pd.Series({            
                    'channel_capacity[min/b]' : minutes_per_timepoint/per_track_empirical['channel_capacity[b/timepoint]'].median(),
                    'channel_capacity_assuming_poisson[min/b]' : minutes_per_timepoint/per_track_empirical['channel_capacity_assuming_poisson[b/timepoint]'].median(),
                })).rename(lambda x: str(x) + '_median_per_track_emp').to_dict(),
                'AVERAGE PER-TRACK PARAMETERS (RESCALED)': '',
                '(compute confusion matrix (CM) \n-> rescale CM to get input frequency as in theory \n -> compute channel capacity \n -> average among cells)': '',
                **per_track_theoretical.drop(['channel_capacity[min/b]', 'channel_capacity_assuming_poisson[min/b]'], axis=1).mean().append(pd.Series({            
                    'channel_capacity[min/b]' : minutes_per_timepoint/per_track_theoretical['channel_capacity[b/timepoint]'].mean(),
                    'channel_capacity_assuming_poisson[min/b]' : minutes_per_timepoint/per_track_theoretical['channel_capacity_assuming_poisson[b/timepoint]'].mean(),
                })).rename(lambda x: str(x) + '_avg_per_track_rescaled').to_dict(),
            }, name=experiment)
            print(output)

            outputs = {**outputs, experiment: output}


            plt.figure('Histograms -- ' + title, figsize=(7,0.6*len(experiments)))
            plt.subplot(len(experiments), 1, it_experiment+1)
            per_track_empirical['channel_capacity[b/h]'].plot.hist(bins=np.linspace(-5-0.1, 20-0.1, 26), alpha=0.5, )
            plt.ylabel(experiment, rotation=0, horizontalalignment='right')
            plt.subplots_adjust(left=0.3)
            



        outputs_df = pd.DataFrame(outputs).T.rename_axis(index='experiment')
        if regular:
            outputs_df['information lost due to inacurate detections[b/timepoint]_emp'] = 0
            outputs_df['information lost due to inacurate detections[b/timepoint]_avg_per_track_emp'] = 0
        
        print(outputs_df)
        if regular: 
            selected_output_fields = outputs_df[['minutes_per_timepoint', 'empirical input period [minutes]', 'min', 'channel_capacity[b/timepoint]_emp',  'information lost due to inacurate detections[b/timepoint]_emp', 'information lost due to false detections[b/timepoint]_emp', 'information lost due to missed pulses[b/timepoint]_emp', 'channel_capacity[b/h]_emp', 'channel_capacity[b/h]_median_per_track_emp']].astype('float').groupby('minutes_per_timepoint').mean().reset_index().sort_values('minutes_per_timepoint', ascending=True).set_index('minutes_per_timepoint', drop=False)
        else:
            selected_output_fields = outputs_df.sort_values('empirical input period [minutes]', ascending=True)
        print(selected_output_fields.T)


        plt.figure(figname)
        selected_output_fields.pipe(lambda x: 60*x[['channel_capacity[b/timepoint]_emp',  'information lost due to inacurate detections[b/timepoint]_emp',  'information lost due to missed pulses[b/timepoint]_emp', 'information lost due to false detections[b/timepoint]_emp']].div(x['minutes_per_timepoint'], axis=0)).pipe(lambda x: print(x.T) or x).pipe(lambda x: x.to_excel(output_path / (figname_without_extension + '--' + parameters['loss_source_determination_method'] + '.xlsx')) or x).pipe(lambda x: x.to_html(output_path / (figname_without_extension + '--' + parameters['loss_source_determination_method'] +'.html'), float_format='{:.2f}'.format) or x).plot.bar(stacked=True, figsize=(4.5,3.5), color=['k', (0., 178/256, 178/256), (87/256, 123/256, 255/256), (189/256, 0., 189/256)], ax=plt.gca())#color=cm.get_cmap('tab10')([0, 1, 3, 2]))#['blue', 'lightskyblue', 'olivedrab', 'lightsalmon']
        plt.ylim(0,26)
        plt.yticks(fontsize=fontsize)
        plt.ylabel('Lost an transmitted information rate [bit/h]', fontsize=fontsize)

        if xaxis == 'empirical input period [minutes]':
            if title == 'Interval encoding with minimal gap':
                plt.xticks(plt.xticks()[0], 
                    [
                        f"{row['empirical input period [minutes]'] :.0f} \n[{row['min']:d}]"  #row['exp_mean']
                            for experiment,row in selected_output_fields.iterrows()
                    ], 
                    rotation=0, fontsize=fontsize)
                plt.xlabel('Mean interval betwean pulses [min]\n[minimal gap $T_{gap}$] [min]', fontsize=fontsize)
            elif title == 'Interval encoding':
                plt.xticks(plt.xticks()[0], 
                    [
                        f"{row['empirical input period [minutes]']:.0f}" 
                            for experiment,row in selected_output_fields.iterrows()
                    ], 
                    rotation=0, fontsize=fontsize)
                plt.xlabel('Mean interval betwean pulses [min]', fontsize=fontsize)
        elif xaxis == 'min':
            plt.xticks(plt.xticks()[0], 
                [
                    f"{row['min']:d} \n[{row['empirical input period [minutes]'] :.0f}]"  #row['exp_mean']
                        for experiment,row in selected_output_fields.iterrows()
                ], 
                rotation=0, fontsize=fontsize)
            plt.xlabel('Minimal gap $T_{gap}$ \n+ [optimal geometric distribution parameter $\\tau$] [min]', fontsize=fontsize)
        elif xaxis == 'minutes_per_timepoint':
            plt.xticks(plt.xticks()[0], [f"{row['minutes_per_timepoint']:.3g}" for _,row in selected_output_fields.iterrows()], rotation=0,  fontsize=fontsize)
            plt.xlabel('Clock period [min]', fontsize=fontsize)

        if not regular:
            plt.legend(reversed(plt.gca().get_legend_handles_labels()[0]), reversed(['transmitted', 'lost due to inacurate detection', 'lost due to missed pulses',  'lost due to false detections']), fontsize='medium')
        else:
            plt.legend(reversed([handle for handle,label in zip(*plt.gca().get_legend_handles_labels()) if label.find('inacurate detection') == -1]), reversed(['transmitted', 'lost due to missed pulses',  'lost due to false detections']), fontsize='medium')

        plt.subplots_adjust(left=0.15, bottom=0.25)
        plt.annotate(figletter, xy=(0.02,.98), xycoords='figure fraction', fontsize='xx-large', verticalalignment='top', fontweight='bold')
        
        if save_each: 
            plt.savefig(output_path / figname)

        

        plt.figure(overall_figname, figsize=(3*6,4))
        plt.subplot(1, 3, group_it+1)

        colors = selected_output_fields.assign(color='blue')['color']#.mask(selected_output_fields.index.isin(mainstream_experiments), 'darkblue')#'deepskyblue'
        print(colors)

        selected_output_fields.plot.scatter(xaxis, 'channel_capacity[b/h]_emp', ax=plt.gca(), c=colors)
        left_of_the_best, right_of_the_best = selected_output_fields[selected_output_fields.index.isin(experiment_manager.best_experiments)][xaxis].agg(['min', 'max'])
        bottom_of_the_best, top_of_the_best = selected_output_fields[selected_output_fields.index.isin(experiment_manager.best_experiments)][ 'channel_capacity[b/h]_emp'].agg(['min', 'max'])
        plt.gca().add_patch(Ellipse(((left_of_the_best+right_of_the_best)/2, (top_of_the_best+bottom_of_the_best)/2), right_of_the_best-left_of_the_best+(8 if not regular else 3), top_of_the_best-bottom_of_the_best+1.5, facecolor=(.6,.6,.6), alpha=0.2, edgecolor='k'))


        xlim = xaxis_to_xlim[xaxis]
        if xaxis == 'empirical input period [minutes]' and title == 'Interval encoding with minimal gap':
            for tmin in selected_output_fields['min'].unique():
                print(optimal_tau(tmin))
                plt.text(tmin+optimal_tau(tmin)+.5, selected_output_fields[selected_output_fields['min'].eq(tmin)]['channel_capacity[b/h]_emp'].max() + .5, f"[{tmin}]", horizontalalignment='center', fontsize='large')
        ylim=(0,10)
        plt.ylim(ylim)
        plt.ylabel('Information transmission rate [bit/h]' if not group_it else '', fontsize='large')
        plt.yticks(fontsize='large')
        plt.xlim(xlim)
        plt.xlabel(xaxis_to_label[xaxis], fontsize='large')
        plt.xticks(fontsize='large')
        plt.grid(True)
        plt.title(title, fontsize='large')
        plt.subplots_adjust(bottom=0.15)

        if regular:
            print(outputs_df[['minutes_per_timepoint', 'channel_capacity[b/h]_emp']].astype('float').groupby('minutes_per_timepoint').std()['channel_capacity[b/h]_emp'] / np.sqrt(outputs_df[['minutes_per_timepoint', 'channel_capacity[b/h]_emp']].astype('float').groupby('minutes_per_timepoint').size() -1))
            if PLOT_ERRORBARS:
                plt.errorbar(selected_output_fields['minutes_per_timepoint'], selected_output_fields['channel_capacity[b/h]_emp'], yerr=outputs_df[['minutes_per_timepoint', 'channel_capacity[b/h]_emp']].astype('float').groupby('minutes_per_timepoint').std()['channel_capacity[b/h]_emp'] / np.sqrt(outputs_df[['minutes_per_timepoint', 'channel_capacity[b/h]_emp']].astype('float').groupby('minutes_per_timepoint').size() -1), fmt='none')


        plt.twinx()
        plt.ylim(ylim)
        plt.yticks(fontsize='large')

        if TICK_BEST_EXPERIMENTS_AVERAGE:
            average_channel_capacity = selected_output_fields[selected_output_fields.index.isin(experiment_manager.best_experiments)]['channel_capacity[b/h]_emp'].mean()
            plt.hlines([average_channel_capacity], *xlim, ls='--', color='mediumblue', alpha=0.5)
            plt.yticks([average_channel_capacity], color='darkblue')
       

        plt.annotate(figletter_overall_figure, xy=(-.1, 1.13), xycoords='axes fraction', fontsize='xx-large', verticalalignment='top', fontweight='bold')

        
        plt.figure(overall_figname + ' -- median', figsize=(3*6,4))
        plt.subplot(1, 3, group_it+1)

        # colors = selected_output_fields.assign(color='blue')['color'].mask(selected_output_fields.index.isin(new_experiments + repeated_experiments), 'blue')#'deepskyblue'
        colors = 'blue'#= selected_output_fields.assign(color='blue')['color']#.mask(selected_output_fields.index.isin(mainstream_experiments), 'darkblue')#'deepskyblue'

        selected_output_fields.plot.scatter(xaxis, 'channel_capacity[b/h]_median_per_track_emp', ax=plt.gca(), c=colors)
        xlim = xaxis_to_xlim[xaxis]
        if xaxis == 'empirical input period [minutes]' and title == 'Interval encoding with minimal gap':
            for tmin in selected_output_fields['min'].unique():
                print(optimal_tau(tmin))
                plt.text(tmin+optimal_tau(tmin)+.5, selected_output_fields[selected_output_fields['min'].eq(tmin)]['channel_capacity[b/h]_median_per_track_emp'].max() + .5, f"[{tmin}]", horizontalalignment='center', fontsize='large')
        ylim=(0,10)
        plt.ylim(ylim)
        plt.ylabel('Information transmission rate [bit/h]' if not group_it else '', fontsize='large')
        plt.yticks(fontsize='large')
        plt.xlim(xlim)
        plt.xlabel(xaxis_to_label[xaxis], fontsize='large')
        plt.xticks(fontsize='large')
        plt.grid(True)
        plt.title(title, fontsize='large')
        plt.subplots_adjust(bottom=0.15)

        if regular:
            print(outputs_df[['minutes_per_timepoint', 'channel_capacity[b/h]_median_per_track_emp']].astype('float').groupby('minutes_per_timepoint').std()['channel_capacity[b/h]_median_per_track_emp'] / np.sqrt(outputs_df[['minutes_per_timepoint', 'channel_capacity[b/h]_median_per_track_emp']].astype('float').groupby('minutes_per_timepoint').size() -1))
            if PLOT_ERRORBARS:
                plt.errorbar(selected_output_fields['minutes_per_timepoint'], selected_output_fields['channel_capacity[b/h]_median_per_track_emp'], yerr=outputs_df[['minutes_per_timepoint', 'channel_capacity[b/h]_median_per_track_emp']].astype('float').groupby('minutes_per_timepoint').std()['channel_capacity[b/h]_median_per_track_emp'] / np.sqrt(outputs_df[['minutes_per_timepoint', 'channel_capacity[b/h]_median_per_track_emp']].astype('float').groupby('minutes_per_timepoint').size() -1), fmt='none')


        plt.twinx()
        plt.ylim(ylim)
        plt.xlim(xlim)


        if not regular:
            outputs_df[[
                'TP_100_emp',
                'TP_010_emp',
                'TP_001_emp',
                'TP_110_emp',
                'TP_101_emp',
                'TP_011_emp',
                'TP_111_emp',
                'FN_000_emp',
                'TN_100_emp',
                'FP_010_emp',
                'TN_001_emp',
                'FP_110_emp',
                'TN_101_emp',
                'FP_011_emp',
                'FP_111_emp',
                'TN_000_emp',
                ]].rename(columns={
                    'TP_100_emp': '100/1',
                    'TP_010_emp': '010/1',
                    'TP_001_emp': '001/1',
                    'TP_110_emp': '110/1',
                    'TP_101_emp': '101/1',
                    'TP_011_emp': '011/1',
                    'TP_111_emp': '111/1',
                    'FN_000_emp': '000/1',
                    'TN_100_emp': '100/0',
                    'FP_010_emp': '010/0',
                    'TN_001_emp': '001/0',
                    'FP_110_emp': '110/0',
                    'TN_101_emp': '101/0',
                    'FP_011_emp': '011/0',
                    'FP_111_emp': '111/0',
                    'TN_000_emp': '000/0',
                }).where(lambda x: x>0).dropna(axis='columns', how='all').pipe(
                        lambda x: x.to_html(output_path / (figname_without_extension + '--3bit_CM.html'), float_format='{:.2%}'.format) or x
                    ).pipe(
                        lambda x: (x.div(outputs_df['input_pulses_emp'], axis=0)).to_html(output_path / (figname_without_extension + '--3bit_CM_normalized.html'), float_format='{:.1%}'.format) or x
                    ).assign(
                        TP=lambda y: y['010/1'],
                        early=lambda y: y['100/1'],
                        late=lambda y: y['001/1'],
                        FN=lambda y: y['000/1'],
                        FP=lambda y: y['010/0'] -  y['001/1'] - y['100/1'],                  
                    ).pipe(
                        lambda x: (x.div(outputs_df['input_pulses_emp'], axis=0))[['TP', 'early', 'late', 'FP', 'FN']].to_html(output_path / (figname_without_extension + '--3bit_CM_interpreted.html'), float_format='{:.2%}'.format) or x
                    )
        else:
            outputs_df[['TP_emp', 'FP_emp', 'FN_emp']].rename(columns=lambda x: x[:-4]).div(outputs_df['input_pulses_emp'], axis=0).to_html(output_path / (figname_without_extension + '--3bit_CM_interpreted.html'), float_format='{:.2%}'.format)

    plt.figure(overall_figname)
    plt.savefig(output_path / overall_figname)
plt.show()

        

    

