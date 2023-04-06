from matplotlib import pyplot as plt
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib.ticker import MultipleLocator, FuncFormatter

sys.path.append(str(Path(__file__).parent.parent))

from core import experiment_manager, factory
from core.steps import get_timeline, scoring_regression_as_yesno, computing_information_transmission_full_3bit_matrix
from core.local_config import full_data_directory

from figures.local_config import figure_output_path
from figures.figure_settings import *
from integrity import check_and_fetch

full_data_directory = Path(full_data_directory).absolute()

check_and_fetch.check_and_fetch_necessary()

output_path = Path(figure_output_path).absolute() / "fig1/automatic/"
output_path.mkdir(parents=True, exist_ok=True)



def is_local_max(x):
    return (x.diff().shift(-1) <= 0) & (x.diff()>0)

def is_local_min(x):
    return is_local_max(-x)



regular = False
experiments = [
    'min3_mean20', 
    'min3_mean30', 
    'min3_mean40',
    'min3_mean20_new', 
    'min3_mean30_new',  
    'min3_mean40_new',  
    ]


t_pos=4

parameters = {
    **experiment_manager.default_parameters,
    'working_directory': full_data_directory,
    'directory': full_data_directory,
    'target_position': t_pos,
    'remove_break': 0,

    **({
        'remove_first_pulses': 0,
        'remove_break': 0,
        'correct_consecutive': 0,
        'remove_shorter_than': 0,
        'yesno': True,
        'n_pulses': 19,
    } if regular else {}),
     
}

chain,prechains = factory.prepare_regular_slices_from_all_experiments(parameters, experiments)

quantified_tracks = chain.load_file('quantified_tracks')
slices = chain.load_file('extracted_slices')

print('Max translocation', end='... ', flush=True)
pwms = 2
timeline = pd.concat(qtrack[qtrack.index >= 90].assign(track_id=track_id).assign(pulse_no=lambda x: (x['isBlink'].cumsum()-1).where(lambda y: y>=0).shift(pwms)) for track_id, qtrack in zip(slices.index.get_level_values('track_id').unique(), quantified_tracks)).rename_axis(index='time').reset_index().set_index(['track_id', 'time'])
pulse_info = pd.DataFrame()
pulse_info['blink'] = timeline.reset_index('time')[['time', 'pulse_no']].groupby(['track_id', 'pulse_no']).min()['time'] - pwms

timeline['is_local_max'] = pd.concat(is_local_max(-track['Q3backw']) for track_id, track in timeline.groupby('track_id'))
pulse_info['maxQ3backwInPulse'] = pd.Series({(track_id,pulse_no): (-pulse['Q3backw']).max() for (track_id,pulse_no), pulse in timeline[timeline['is_local_max']].groupby(['track_id', 'pulse_no'])})
pulse_info['argmaxQ3backwInPulse'] = pd.Series({(track_id,pulse_no): (-pulse['Q3backw']).idxmax()[1] for (track_id,pulse_no), pulse in timeline[timeline['is_local_max']].groupby(['track_id', 'pulse_no'])}) 
pulse_info['argmaxQ3backwInPulse'] = pulse_info['argmaxQ3backwInPulse']- pulse_info['blink']

print('Max increment', end='... ', flush=True)


pwms = 1
timeline = pd.concat(qtrack[qtrack.index >= 90].assign(track_id=track_id).assign(pulse_no=lambda x: (x['isBlink'].cumsum()-1).where(lambda y: y>=0).shift(pwms)) for track_id, qtrack in zip(slices.index.get_level_values('track_id').unique(), quantified_tracks)).rename_axis(index='time').reset_index().set_index(['track_id', 'time'])

timeline['is_local_max'] = pd.concat(is_local_max(-track['dQ3backw']) for track_id, track in timeline.groupby('track_id'))
pulse_info['maxdQ3backwInPulse'] = pd.Series({(track_id,pulse_no): (-pulse['dQ3backw']).max() for (track_id,pulse_no), pulse in timeline[timeline['is_local_max']].groupby(['track_id', 'pulse_no'])})
pulse_info['argmaxdQ3backwInPulse'] = pd.Series({(track_id,pulse_no): (-pulse['dQ3backw']).idxmax()[1] for (track_id,pulse_no), pulse in timeline[timeline['is_local_max']].groupby(['track_id', 'pulse_no'])}) 
pulse_info['argmaxdQ3backwInPulse'] -= pulse_info['blink']

print('Max decrement', end='... ', flush=True)

pwms = 3
timeline = pd.concat(qtrack[qtrack.index >= 90].assign(track_id=track_id).assign(pulse_no=lambda x: (x['isBlink'].cumsum()-1).where(lambda y: y>=0).shift(pwms)) for track_id, qtrack in zip(slices.index.get_level_values('track_id').unique(), quantified_tracks)).rename_axis(index='time').reset_index().set_index(['track_id', 'time'])

timeline['is_local_min'] = pd.concat(is_local_min(-track['dQ3backw']) for track_id, track in timeline.groupby('track_id'))
pulse_info['mindQ3backwInPulse'] = pd.Series({(track_id,pulse_no): (-pulse['dQ3backw']).min() for (track_id,pulse_no), pulse in timeline[timeline['is_local_min']].groupby(['track_id', 'pulse_no'])})
pulse_info['argmindQ3backwInPulse'] = pd.Series({(track_id,pulse_no): (-pulse['dQ3backw']).idxmin()[1] for (track_id,pulse_no), pulse in timeline[timeline['is_local_min']].groupby(['track_id', 'pulse_no'])}) 
pulse_info['argmindQ3backwInPulse'] -= pulse_info['blink']

print('Detections', end='... ', flush=True)

prechains = factory.for_each_experiment(factory.detecting_blink_regr, parameters, experiments)
chain2 = factory.combining_results(parameters, prechains).step(scoring_regression_as_yesno).step(get_timeline).step(computing_information_transmission_full_3bit_matrix)
binary_timeline2 = chain2.load_file('binary_timeline')
results2 = chain2.load_file('prediction_results')


pwms = 2
TAP = pd.concat([x['input_blinks'].mul(0).add(1).cumsum().subtract(
    x['input_blinks'].mul(0).add(x['input_blinks'].mul(0).add(1).cumsum()[x['input_blinks']==1]).fillna(method='pad').shift(pwms-t_pos)
    ) for ind, x in  binary_timeline2.groupby(['experiment', 'classifier', 'iteration', 'track_id'])])
TAP.name = 'TAP'
print(TAP.head(150).tail(40))
detections = binary_timeline2[binary_timeline2['output_detections'] >=.5].join(TAP).join(results2['pulse_no']).reset_index().set_index(['experiment', 'track_id', 'pulse_no'])['TAP']
print(binary_timeline2[binary_timeline2['output_detections'] >=.5].join(TAP).join(results2['pulse_no']))
print(binary_timeline2[binary_timeline2['output_detections'] >=.5])
print(detections)




# if experiment == 'min3_mean30':
#     pulse_info = pulse_info[pulse_info['pulse_no']>=90]


def plot_field(field, label, offset, color, xticks):
    (pulse_info[field]-offset).plot.hist(bins=np.array(range(40))-0.5-offset, density=True, alpha=0.6, color=color)
    # print(pulse_info[field].pipe(lambda x: x.groupby(x).count()))
    plt.gca().spines['top'].set(visible=False)  
    plt.gca().spines['right'].set(visible=False)
    plt.ylabel('% of pulses', fontsize='large')
    plt.text(15, (.3*plt.ylim()[0]+.7*plt.ylim()[1]), label, fontsize='large')
    plt.xticks(xticks, fontsize='large')
    plt.yticks(fontsize='large')
    plt.gca().xaxis.set_minor_locator(MultipleLocator(10))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0%}"))
plt.imread


plt.figure(figsize=(3,3.8))
plt.subplot(3, 1, 1)
plot_field('argmaxdQ3backwInPulse', 'Max increment',  0.5, 'blue', xticks=[0, 3, 40])

# (pulse_info['argmaxdQ3backwInPulse']-0.5).plot.hist(bins=np.array(range(40))-1, density=True, alpha=0.6, color='blue')
# print(pulse_info['argmaxdQ3backwInPulse'].pipe(lambda x: x.groupby(x).count()))
# plt.gca().spines['top'].set(visible=False)
# plt.gca().spines['right'].set(visible=False)
# plt.ylabel('% of pulses', fontsize='large')
# plt.text(15, (.3*plt.ylim()[0]+.7*plt.ylim()[1]), 'Max increment', fontsize='large')
# plt.xticks([0, 3, 40], fontsize='large')
# plt.yticks(fontsize='large')
# plt.gca().xaxis.set_minor_locator(MultipleLocator(10))
# plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0%}"))

plt.subplot(3, 1, 2)
plot_field('argmaxQ3backwInPulse', 'Max translocation', 0, 'orange', xticks=[0, 6, 40])

# pulse_info['argmaxQ3backwInPulse'].plot.hist(bins=np.array(range(40))-0.5, density=True, alpha=0.6, color='orange')
# print(pulse_info['argmaxQ3backwInPulse'].pipe(lambda x: x.groupby(x).count()))
# plt.gca().spines['top'].set(visible=False)
# plt.gca().spines['right'].set(visible=False)
# plt.xticks([0, 6, 40], fontsize='large')
# plt.text(15, (.3*plt.ylim()[0]+.7*plt.ylim()[1]), 'Max translocation', fontsize='large')
# plt.yticks(fontsize='large')
# plt.ylabel('% of pulses', fontsize='large')
# plt.gca().xaxis.set_minor_locator(MultipleLocator(10))
# plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0%}"))

plt.subplot(3, 1, 3)

plot_field('argmindQ3backwInPulse', 'Max decrement', 0.5, 'green', xticks=[0, 11, 40])
# (pulse_info['argmindQ3backwInPulse']-0.5).plot.hist(bins=np.array(range(40))-1, density=True, alpha=0.6, color='green')
# print(pulse_info['argmindQ3backwInPulse'].pipe(lambda x: x.groupby(x).count()))
# plt.gca().spines['top'].set(visible=False)
# plt.gca().spines['right'].set(visible=False)
# plt.text(15, (.3*plt.ylim()[0]+.7*plt.ylim()[1]), 'Max decrement', fontsize='large')
# plt.yticks(fontsize='large')
# plt.ylabel('% of pulses', fontsize='large')
# plt.xticks([0, 11, 40], fontsize='large')
# plt.gca().xaxis.set_minor_locator(MultipleLocator(10))
# plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0%}"))

plt.xlabel('Time after pulse [min]', fontsize='large')

plt.subplots_adjust(left=0.35, hspace=.33, bottom=0.2, right=0.83)

plt.annotate("E", xy=(0.05,.98), xycoords='figure fraction', fontsize='xx-large', verticalalignment='top', fontweight='bold')
plt.savefig(output_path / 'fig1E.svg')

plt.figure(figsize=(3,1.4))
detections.plot.hist(bins=np.array(range(-20, 20))-0.5, density=True, alpha=0.6)
plt.gca().spines['top'].set(visible=False)
plt.gca().spines['right'].set(visible=False)
plt.text(-20, (-.05*plt.ylim()[0]+1.05*plt.ylim()[1]), 'Pulse detection precision', fontsize='large')
plt.ylabel('% of pulses', fontsize='large')
plt.xlabel('$t_{predicted} - t_{true}$ [min]', fontsize='large')
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0%}"))
plt.ylim(0,1)

plt.subplots_adjust(left=.35, bottom=0.35, top=0.95)

plt.annotate("F", xy=(0.05,1.23), xycoords='figure fraction', fontsize='xx-large', verticalalignment='top', fontweight='bold')
plt.savefig(output_path / 'fig1F.svg')

print(pulse_info.groupby(['argmaxdQ3backwInPulse', 'argmaxQ3backwInPulse']).size().pipe(lambda x: x[(x.index.get_level_values('argmaxdQ3backwInPulse') < 7) & (x.index.get_level_values('argmaxQ3backwInPulse') < 7)]))



print(timeline)
print(pulse_info)
print(pulse_info[['argmaxdQ3backwInPulse', 'argmaxQ3backwInPulse', 'argmindQ3backwInPulse']].describe())
print(pulse_info[['argmaxdQ3backwInPulse', 'argmaxQ3backwInPulse', 'argmindQ3backwInPulse']].pipe(lambda x : x[x<20]).describe())
print(pulse_info[['argmaxdQ3backwInPulse', 'argmaxQ3backwInPulse', 'argmindQ3backwInPulse']].pipe(lambda x : x[x<40]).describe())
print(pulse_info[['maxdQ3backwInPulse', 'maxQ3backwInPulse', 'mindQ3backwInPulse']].describe())
print(detections.describe())
print(detections.pipe(lambda x : x[x<20]).describe())
print(pulse_info['argmaxdQ3backwInPulse'].pipe(lambda x : x[x<9]).describe())
print(pulse_info['argmaxQ3backwInPulse'].pipe(lambda x : x[x<14]).describe())
print(pulse_info['argmindQ3backwInPulse'].pipe(lambda x : x[x<24]).describe())


plt.figure()
detections.dropna().groupby(['track_id', 'pulse_no']).size().reindex(pulse_info.index).fillna(0).plot.hist(bins=range(-1, 10), density=True)
print('MISSED:')
print(pulse_info[['argmaxdQ3backwInPulse', 'argmaxQ3backwInPulse', 'argmindQ3backwInPulse']].isna().mean())
# print('detections', 1- len(detections.reset_index()[['experiment', 'track_id', 'pulse_no', 'iteration']].unique()) / len(binary_timeline2.reset_index()[['experiment', 'track_id', 'pulse_no', 'iteration']].unique()))

plt.show()

