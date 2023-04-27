import sys
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from matplotlib.ticker import MultipleLocator, FuncFormatter
import matplotlib.patheffects as PathEffects

sys.path.append(str(Path(__file__).parent.parent))

from core import experiment_manager, factory

from core.local_config import root_working_directory
from figures.local_config import figure_output_path
from figures.figure_settings import *

from utils.data_utils import index_without
from integrity import check_and_fetch

check_and_fetch.check_and_fetch_necessary()

output_path = Path(figure_output_path).absolute() / "fig1"
output_path.mkdir(parents=True, exist_ok=True)
root_working_directory = Path(root_working_directory).absolute()


experiments = experiment_manager.chosen_experiments_interval 

parameters = {
    **experiment_manager.default_parameters,
    'working_directory': root_working_directory,
    'directory': root_working_directory,
}


chains = factory.for_each_experiment(factory.compute_information_transmission(regular=False, learning=True), parameters, experiment_list=experiments, )

def get_detection_TAP(experiment):
    chain = chains[experiment]
    binary_predictions: pd.DataFrame = chain.load_file('binary_predictions')
    blinks: pd.Series = chain.load_file('blinks')
    blinks.name = 'last_blink'

    previous_pulse_lengths: pd.Series = chain.load_file('previous_pulse_lengths')

    detections_per_pulse = pd.DataFrame()
    detections_per_pulse[' '] = 0
    detections_per_pulse['on time']         = (binary_predictions['y_true'] * binary_predictions['y_pred'].groupby(index_without(binary_predictions, ['time_point'])).shift(0) )[binary_predictions['y_true'] == 1].to_frame().join(binary_predictions['pulse_no']).groupby('pulse_no').mean()
    detections_per_pulse['2 min too early'] = (binary_predictions['y_true'] * binary_predictions['y_pred'].groupby(index_without(binary_predictions, ['time_point'])).shift(2) )[binary_predictions['y_true'] == 1].to_frame().join(binary_predictions['pulse_no']).groupby('pulse_no').mean()
    detections_per_pulse['1 min too early'] = (binary_predictions['y_true'] * binary_predictions['y_pred'].groupby(index_without(binary_predictions, ['time_point'])).shift(1) )[binary_predictions['y_true'] == 1].to_frame().join(binary_predictions['pulse_no']).groupby('pulse_no').mean()
    detections_per_pulse['1 min too late']  = (binary_predictions['y_true'] * binary_predictions['y_pred'].groupby(index_without(binary_predictions, ['time_point'])).shift(-1))[binary_predictions['y_true'] == 1].to_frame().join(binary_predictions['pulse_no']).groupby('pulse_no').mean()
    detections_per_pulse['2 min too late']  = (binary_predictions['y_true'] * binary_predictions['y_pred'].groupby(index_without(binary_predictions, ['time_point'])).shift(-2))[binary_predictions['y_true'] == 1].to_frame().join(binary_predictions['pulse_no']).groupby('pulse_no').mean()

    return detections_per_pulse.join(previous_pulse_lengths, on='pulse_no')


detections_per_experiment_and_pulse = pd.concat([get_detection_TAP(experiment) for experiment in experiments], keys=experiments, names=['experiment'])   
print(detections_per_experiment_and_pulse)

plt.figure('fig1G', figsize=(7,5.2))

detections_per_ppl = detections_per_experiment_and_pulse.groupby('previous_pulse_length').mean()
(lambda x: x[['on time', ' ', '1 min too early', '1 min too late', '2 min too early', '2 min too late']]\
            .plot.bar(
                stacked=True, 
                bottom=-0*(x['2 min too early']/2 + x['1 min too early']/2 + x['on time']/2 + x['1 min too late']/2 + x['2 min too late']/2).fillna(0),
                color=['green', 'white', 'orange', 'darkgoldenrod', 'wheat', 'tan'],
                ax=plt.gca())
            )(detections_per_ppl.reindex(range(100))) #['slateblue', 'deepskyblue', 'green', 'orange', 'red']['green', 'orange', 'darkgoldenrod', 'wheat', 'tan']['on time', '1 min too early', '1 min too late', '2 min too early', '2 min too late']


plt.xticks(fontweight='bold', color='k', alpha=1, rotation=0, path_effects=[PathEffects.withStroke(linewidth=5, foreground='w')], fontsize='large')
plt.xlabel('Time after previous pulse [min]', path_effects=[PathEffects.withStroke(linewidth=7, foreground='w')],  alpha=1, fontsize='large') #bbox=dict(color='white', alpha=0.6),

plt.ylim((0,1.001))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0%}"))
plt.yticks(fontsize='large')
plt.ylabel('Pulse detection [% of cells]', fontsize='large')

plt.grid(ls=':')
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
plt.gca().spines['top'].set(visible=False)
plt.gca().spines['right'].set(visible=False)
plt.gca().axes.set(axisbelow = False)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
          ncol=3, fancybox=True, shadow=True, fontsize='large')
plt.subplots_adjust(top=0.85)

plt.annotate("G", xy=(0.02,.98), xycoords='figure fraction', fontsize='xx-large', verticalalignment='top', fontweight='bold')
plt.savefig(output_path / 'fig1G.svg')


detections_per_experiment_and_pulse.to_csv(figure_output_path / 'fig1G.csv')


plt.show()


