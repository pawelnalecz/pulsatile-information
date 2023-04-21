from matplotlib import pyplot as plt
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib.ticker import MultipleLocator, FuncFormatter

sys.path.append(str(Path(__file__).parent.parent))

from core import experiment_manager, factory
from core.local_config import full_data_directory

from utils.data_utils import index_without
from utils.math_utils import is_local_max, is_local_min

from figures.local_config import figure_output_path
from figures.figure_settings import *
from integrity import check_and_fetch

full_data_directory = Path(full_data_directory).absolute()

check_and_fetch.check_and_fetch_necessary()

output_path = Path(figure_output_path).absolute() / "fig1"
output_path.mkdir(parents=True, exist_ok=True)


experiments = experiment_manager.chosen_experiments_interval

parameters = {
    **experiment_manager.default_parameters,
    'working_directory': full_data_directory,
    'directory': full_data_directory,
    # 'pulse_window_matching_shift': 0,
}

chain,prechains = factory.prepare_slices_from_all_experiments(parameters, experiments)

quantified_tracks = chain.load_file('quantified_tracks')

pulse_info = pd.DataFrame()

timeline = pd.concat((qtrack.loc[90:].rename_axis(index='time_point') for qtrack in quantified_tracks), names=['track_id'], keys=list(range(len(quantified_tracks))))
timeline['pulse_no'] = timeline.groupby(index_without(timeline, ['time_point']))['isBlink'].cumsum() - 1
timeline['pulse_no'] = timeline['pulse_no'].where(timeline['pulse_no'] >= 0)

blinks = timeline.reset_index('time_point')[['time_point', 'pulse_no']].groupby(['track_id', 'pulse_no']).min()['time_point']
blinks.name = 'blinks'
pulse_info['blinks'] = blinks



def find_extrema_for_each_pulse(field, kind, pwms):
    assert kind in ('min', 'max')
    is_local_extremum = is_local_max if kind == 'max' else is_local_min

    timeline_shifted = timeline.copy()
    timeline_shifted['pulse_no'] = timeline_shifted.groupby(index_without(timeline_shifted, ['time_point']))['pulse_no'].shift(pwms)
    timeline_shifted = timeline_shifted.set_index('pulse_no', append=True)
    timeline_shifted['is_local_extremum'] = pd.concat(is_local_extremum(-track[field]) for track_id, track in timeline_shifted.groupby('track_id'))
    
    pulse_info_tmp = (-timeline_shifted[timeline_shifted['is_local_extremum']]).groupby(['track_id', 'pulse_no'])[field].agg(['size', kind, 'idx' + kind])
    pulse_info[field, kind] = pulse_info_tmp[kind]
    pulse_info[field, 'arg' + kind] = pulse_info_tmp['idx' + kind].apply(lambda x: x[1]) - pulse_info['blinks']


def plot_field(field, label, offset, color, xticks):
    (pulse_info[field]-offset).dropna().plot.hist(bins=np.array(range(40))-0.5-offset, density=True, alpha=0.6, color=color)
    plt.gca().spines['top'].set(visible=False)
    plt.gca().spines['right'].set(visible=False)
    plt.ylabel('% of pulses', fontsize='large')
    plt.text(15, (.3*plt.ylim()[0]+.7*plt.ylim()[1]), label, fontsize='large')
    plt.xticks(xticks, fontsize='large')
    plt.yticks(fontsize='large')
    plt.gca().xaxis.set_minor_locator(MultipleLocator(10))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0%}"))




plt.figure(figsize=(3,3.8))

print('Max translocation', end='... ', flush=True)
plt.subplot(3, 1, 1)
find_extrema_for_each_pulse('dQ3backw', kind='max', pwms=1)
plot_field(('dQ3backw', 'argmax'), 'Max increment',  0.5, 'blue', xticks=[0, 3, 40])


print('Max increment', end='... ', flush=True)
plt.subplot(3, 1, 2)
find_extrema_for_each_pulse('Q3backw', kind='max', pwms=2)
plot_field(('Q3backw', 'argmax'), 'Max translocation', 0, 'orange', xticks=[0, 6, 40])


print('Max decrement', end='... ', flush=True)
plt.subplot(3, 1, 3)
find_extrema_for_each_pulse('dQ3backw', kind='min', pwms=3)
plot_field(('dQ3backw', 'argmin'), 'Max decrement', 0.5, 'green', xticks=[0, 11, 40])



plt.xlabel('Time after pulse [min]', fontsize='large')
plt.subplots_adjust(left=0.35, hspace=.33, bottom=0.2, right=0.83)

plt.annotate("E", xy=(0.05,.98), xycoords='figure fraction', fontsize='xx-large', verticalalignment='top', fontweight='bold')
plt.savefig(output_path / 'fig1E.svg')



print('Detections', end='... ', flush=True)

prechains = factory.for_each_experiment(factory.compute_information_transmission(regular=False, learning=True), parameters, experiment_list=experiments, )

def get_detection_TAP(experiment, pwms):
    chain = prechains[experiment]
    binary_predictions: pd.DataFrame = chain.load_file('binary_predictions')
    blinks: pd.Series = chain.load_file('blinks')
    blinks.name = 'last_blink'
    binary_predictions['pulse_no'] = binary_predictions.groupby(index_without(binary_predictions, ['time_point']))['pulse_no'].shift(pwms - parameters['pulse_window_matching_shift'])
    binary_predictions = binary_predictions.join(blinks, on='pulse_no')
    binary_predictions = binary_predictions.assign(TAP=lambda x: x.index.get_level_values('time_point') - x['last_blink'])
    return binary_predictions[binary_predictions['y_pred'] == 1].set_index('pulse_no', append=True)['TAP']


detections = pd.concat([get_detection_TAP(experiment, pwms=2) for experiment in experiments], keys=experiments, names=['experiments'])

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


print(detections)

plt.figure()
detections.dropna().groupby(['track_id', 'pulse_no']).size().reindex(pulse_info.index).fillna(0).plot.hist(bins=range(-1, 10), density=True)

plt.show()

