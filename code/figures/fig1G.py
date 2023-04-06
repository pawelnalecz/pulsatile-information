import sys
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from matplotlib.ticker import MultipleLocator, FuncFormatter
import matplotlib.patheffects as PathEffects

sys.path.append(str(Path(__file__).parent.parent))

from core import experiment_manager, factory
from core.steps import computing_information_transmission_full_3bit_matrix, get_timeline, computing_information_transmission

from figures.local_config import figure_output_path
from figures.figure_settings import *
from integrity import check_and_fetch

check_and_fetch.check_and_fetch_necessary()

output_path = Path(figure_output_path).absolute() / "fig1/automatic/"
output_path.mkdir(parents=True, exist_ok=True)


t_pos=4
regular =False

experiments = [
    'min3_mean20', 
    'min3_mean30', 
    'min3_mean40',
    'min3_mean20_new', 
    'min3_mean30_new',  
    'min3_mean40_new',  
    ]

experiments_pseudorandom = (
            [f'pseudorandom_pos{pos:02d}_period{period:d}' for period in [10, 15] for pos in range(1,11) ]+
            [f'pseudorandom_pos{pos:02d}_period{period:d}' for period in [20, 30] for pos in range(1,11) ]+
            [f'pseudorandom_pos{pos:02d}_period{period:d}' for period in [5, 7] for pos in range(2,10) ]+
            [f'pseudorandom_pos{pos:02d}_period{period:d}_new' for period in [3, 10, 15] for pos in range(1,11)]
            )
if regular:
    experiments = experiments_pseudorandom

parameters = {
    **experiment_manager.default_parameters,
    'remove_break': 0,
}


prechains = factory.for_each_experiment(factory.detecting_blink_regr if not regular else factory.deciding_if_pulse, parameters, experiments)
chain = factory.combining_results(parameters, prechains).step(get_timeline).step(computing_information_transmission_full_3bit_matrix if not regular else computing_information_transmission)


results : pd.DataFrame = chain.load_file('prediction_results')
binary_timeline = chain.load_file('binary_timeline')
information_per_track_empirical : pd.DataFrame = chain.load_file('information_per_track_empirical')


results = results[results.index.get_level_values('slice_no') >=90]
binary_timeline = binary_timeline[binary_timeline.index.get_level_values('slice_no') >=90]


previous_pulse_lengths = pd.concat(pd.DataFrame(prechains[experiment].load_file('previous_pulse_lengths')).assign(experiment=experiment).reset_index().rename(columns={'index': 'pulse_no'}).set_index(['experiment', 'pulse_no'])['previous_pulse_length'] for experiment in experiments)

pulse_nos = binary_timeline.groupby(['classifier', 'iteration', 'track_id']).apply(lambda x: x[x['input_blinks'] == 1])
print(pulse_nos)

pulse_nos = results.groupby(['experiment', 'slice_no']).mean()['pulse_no'].sort_index()

info_per_pulse = pd.DataFrame()
info_per_pulse[' '] = 0
info_per_pulse['on time'] = binary_timeline.assign(shifted_output_detecions=lambda x: x.groupby('track_id')['output_detections'].shift(0)).pipe(lambda x: x[x['input_blinks']==1]).join(pulse_nos, on=['experiment', 'slice_no']).groupby(['experiment', 'pulse_no']).mean()['shifted_output_detecions']
info_per_pulse['2 min too early'] = 0 if regular else binary_timeline.assign(shifted_output_detecions=lambda x: x.groupby('track_id')['output_detections'].shift(2)).pipe(lambda x: x[x['input_blinks']==1]).join(pulse_nos, on=['experiment', 'slice_no']).groupby(['experiment', 'pulse_no']).mean()['shifted_output_detecions']
info_per_pulse['1 min too early'] = 0 if regular else binary_timeline.assign(shifted_output_detecions=lambda x: x.groupby('track_id')['output_detections'].shift(1)).pipe(lambda x: x[x['input_blinks']==1]).join(pulse_nos, on=['experiment', 'slice_no']).groupby(['experiment', 'pulse_no']).mean()['shifted_output_detecions']
info_per_pulse['1 min too late'] = 0 if regular else binary_timeline.assign(shifted_output_detecions=lambda x: x.groupby('track_id')['output_detections'].shift(-1)).pipe(lambda x: x[x['input_blinks']==1]).join(pulse_nos, on=['experiment', 'slice_no']).groupby(['experiment', 'pulse_no']).mean()['shifted_output_detecions']
info_per_pulse['2 min too late'] = 0 if regular else binary_timeline.assign(shifted_output_detecions=lambda x: x.groupby('track_id')['output_detections'].shift(-2)).pipe(lambda x: x[x['input_blinks']==1]).join(pulse_nos, on=['experiment', 'slice_no']).groupby(['experiment', 'pulse_no']).mean()['shifted_output_detecions']


plt.figure(figsize=(7,5.2))

print(info_per_pulse)
info_per_ppl = info_per_pulse.join(previous_pulse_lengths).groupby('previous_pulse_length').mean()
(lambda x: print(x) or x[['on time', ' ', '1 min too early', '1 min too late', '2 min too early', '2 min too late']].plot.bar(stacked=True, bottom=-0*(x['2 min too early']/2 + x['1 min too early']/2 + x['on time']/2 + x['1 min too late']/2 + x['2 min too late']/2).fillna(0), color=['green', 'white', 'orange', 'darkgoldenrod', 'wheat', 'tan'], ax=plt.gca()))(info_per_ppl.reindex(range(100))) #['slateblue', 'deepskyblue', 'green', 'orange', 'red']['green', 'orange', 'darkgoldenrod', 'wheat', 'tan']['on time', '1 min too early', '1 min too late', '2 min too early', '2 min too late']

print(info_per_ppl.reindex(range(10,16)))


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


info_per_pulse.groupby('experiment').mean().plot.bar()


# previous_pulse_lengths.pipe(lambda x: x.groupby(x).size()).reindex(range(100)).plot.bar()
# results.join(previous_pulse_lengths, on=['experiment', 'pulse_no']).groupby('previous_pulse_length').size().reindex(range(100)).plot.bar()
# print(results.join(previous_pulse_lengths, on=['experiment', 'pulse_no']).groupby('previous_pulse_length').size().pipe(lambda x: x*x.index).mean())





print(info_per_pulse)

plt.show()


