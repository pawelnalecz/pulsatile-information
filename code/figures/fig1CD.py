from matplotlib import pyplot as plt
import sys
import numpy as np
import pandas as pd
from matplotlib import patches

from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.utils import draw_contour_gen, show_cell_in_image_gen
from core import experiment_manager, factory
from core.local_config import full_data_directory, images_root_directory, external_data_directory, DATA_SOURCE
from figures.local_config import figure_output_path
from integrity import check_and_fetch

full_data_directory = Path(full_data_directory).absolute()
images_root_directory = Path(images_root_directory).absolute()
external_data_directory = Path(external_data_directory).absolute()

check_and_fetch.check_and_fetch_necessary(requires_fig1cd_package=True)

output_path = Path(figure_output_path).absolute() / "fig1"
output_path.mkdir(parents=True, exist_ok=True)


tab10 = plt.get_cmap('tab10')
used_cmap = lambda x: 'maroon' if x == 7 else tab10(x) if x < 10 else ['lime', 'magenta'][x-10]


# experiment, N, neighboring_cells, pulse_no, onset, channel = 'min3_mean30', 27, [474, 583, 904, 935, 259, 476, 648, 649, 259, 59, 631, 244, 521, 370, 720,], 109,0,1
experiment, N, neighboring_cells, pulse_no, onset, channel = 'min3_mean30', 264, [151, 345, 912, 363, 430, 260, 671, 693, 315, 761, 229, 374], 109, 0, 1
#concider N= 264 or 205


parameters = {
    **experiment_manager.default_parameters,
    **experiment_manager.experiments[experiment],
    'theoretical_parameters': experiment_manager.theoretical_parameters[experiment],
    **({
        'working_directory': images_root_directory.joinpath(experiment_manager.get_official_directory(experiment).parent.relative_to(external_data_directory)),
        'directory': experiment_manager.map_to_official_naming(experiment),
    } if DATA_SOURCE == 'EXTERNAL' else {}),
    'trim_end': experiment_manager.trim_end(experiment),
    'n_tracks': 'None',
    'correct_consecutive': 2,
                
    'vivid_track_criteria': [
        ('', 'index', 'lt', 500),
        ('std_dQ2', 'rank', 'gt', 0.2),
    ],
}



chain = factory.compute_information_transmission_using_reconstruction(parameters)

blinks = chain.load_file('blinks')
quantified_tracks = chain.load_file('quantified_tracks')
previous_pulse_lengths = chain.load_file('previous_pulse_lengths')
binary_predictions = chain.load_file('binary_predictions')


show_cell = show_cell_in_image_gen(chain,
    ['']*channel + [plt.get_cmap('Greys_r'), '', ''],
    scatter_dict=dict(s=0)
)
draw_contour = draw_contour_gen(chain)


# Fig 1C
Ns = neighboring_cells
timepoints = range(-1, 15, 1)#range(-1, 23, 1)
rows=2#3
plt.figure(figsize=(1.25 * len(timepoints) / rows, 1.5 + rows))
for i,T in enumerate(timepoints):
    plt.subplot(rows, int(np.ceil(len(timepoints)/rows)), i+1)
    x0,y0 = show_cell(N, blinks[pulse_no] + T, channels=[channel], norm_quantiles=(.1,.9), window_half_size=60, shift_in_time=onset)
    xlim = plt.xlim()
    ylim = plt.ylim()
    draw_contour(N, blinks[pulse_no] + T, plotdict=dict(color='blue', lw=1))
    for it, track_id in enumerate(Ns):
        draw_contour(track_id, blinks[pulse_no] + T, plotdict=dict(color=used_cmap(it), lw=1))
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.axis('off')
    plt.title("$\\blacktriangledown$" if (blinks[pulse_no] + T) in blinks.tolist() else f"$t_0{T:+d}$ min" if T else "$t_0$", fontdict=dict(color='dodgerblue', size='xx-large') if (blinks[pulse_no] + T) in blinks.tolist() else {})
    if (blinks[pulse_no] + T) in blinks.tolist():
        print(T)
        plt.gca().spines['left'].set(visible=True, color='k', lw=3, ls='-')
        plt.gca().add_patch(patches.Rectangle((0,0), 80, 80, color='red'))

    else: print(f"{blinks[pulse_no] + T:d} not in blinks")


plt.annotate("C", xy=(0.02,.98), xycoords='figure fraction', fontsize='xx-large', verticalalignment='top', fontweight='bold')
plt.subplots_adjust(left=0.05, right=0.95)
plt.savefig(output_path / 'fig1C.svg')


# Fig 1D
Q = chain.load_file('raw_tracks')
assert all(track_id < len(Q) for track_id in Ns), f"To generate fig 1D, all cells must be present in the raw track file" + (". Try switching DATA_SOURCE to 'INTERNAL' in core.local_config" if DATA_SOURCE == 'EXTERNAL' else '')

plt.figure(figsize=(10,4))

xlim  = (blinks[pulse_no-1]-2, blinks[pulse_no+1]+30)

ylim = (-0.3, 0.5)
plt.gca().add_patch(patches.Rectangle((blinks[pulse_no] + timepoints[0], ylim[0]),  timepoints[-1] - timepoints[0], ylim[1]-ylim[0], facecolor='brown', alpha=0.15))

(1-quantified_tracks[N]['Q3backw']).loc[xlim[0]:xlim[1]].plot()

for it,track_id in enumerate(Ns):
        (1-quantified_tracks[track_id]['Q3backw']).loc[xlim[0]:xlim[1]].plot(alpha=0.3, color=used_cmap(it), ls='-')
   

plt.xticks(fontsize='large')
plt.xlim(xlim)


yticks = np.array([-0.2, 0, 0.2, 0.4])
plt.yticks(yticks, fontsize='large')
plt.ylim(ylim)
plt.ylabel('ERK KTR translocation [a.u.]', fontsize='large') 

plt.xticks(*zip(*[(blinks[pulse_no] + t, f"$t_0{t:+d}$ min" if t else "$t_0$") for t in [-40, -20, 0, 20, 40]]))
plt.xlabel('Time', fontsize='large')

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

# auxiliary axis

ax1 = plt.gca()
ax2 = plt.gca().twinx()
ax1.set(zorder=ax2.get_zorder()+1)
ax1.patch.set_visible(False)
def plot_percentage(df: pd.DataFrame, ):
    plotter = lambda x, color: plt.bar(x.index.get_level_values('time_point'), x['y_pred'], color=color, alpha=0.8)
    plotter(df[(df.index.get_level_values('time_point') - 2).isin(blinks)], 'tan')
    plotter(df[(df.index.get_level_values('time_point') - 1).isin(blinks)], 'darkgoldenrod')
    plotter(df[(df.index.get_level_values('time_point')    ).isin(blinks)], 'green')
    plotter(df[(df.index.get_level_values('time_point') + 1).isin(blinks)], 'orange')
    plotter(df[(df.index.get_level_values('time_point') + 2).isin(blinks)], 'wheat')
    plotter(df[~(
        (  df.index.get_level_values('time_point') - 2).isin(blinks)
        | (df.index.get_level_values('time_point') - 1).isin(blinks)
        | (df.index.get_level_values('time_point')    ).isin(blinks)
        | (df.index.get_level_values('time_point') + 1).isin(blinks)
        | (df.index.get_level_values('time_point') + 2).isin(blinks))], 'grey')
    return df
binary_predictions[binary_predictions.index.get_level_values('time_point').isin(range(xlim[0], xlim[1]))].groupby(['time_point']).mean().pipe(plot_percentage)
for time_point, row in (lambda x: x[x>0.5])(binary_predictions.assign(pm2_detections = lambda x: x['y_pred'].rolling(5, center=True).sum())[binary_predictions.index.get_level_values('time_point').isin(range(*xlim))].groupby(['time_point']).mean()).iterrows():
    plt.annotate(f"{100*row['y_pred']:.0f}%", (time_point, row['y_pred']), horizontalalignment='left', xytext=(0,4), textcoords='offset points', color='green')
plt.ylim(0,1)
plt.ylabel('Pulse detection [% of cells]', fontsize='large')
plt.yticks([0, 0.25, .5, .75, 1], ['0%', '25%', '50%', '75%', '100%', ], fontsize='large')

plt.plot(*zip(*[ (x, .97) for x in blinks[blinks.isin(range(int(np.ceil(plt.xlim()[0])), int(np.ceil(plt.xlim()[1]))))]]), ls='none', marker='v', color='dodgerblue', ms='8')
for blink,previous_pulse_length in zip(blinks.loc[pulse_no:(pulse_no+2)],previous_pulse_lengths.loc[pulse_no:(pulse_no+2)]):
    plt.annotate(f'{previous_pulse_length:.0f} min', (blink-previous_pulse_length/2, .98), fontweight='bold', horizontalalignment='center', verticalalignment='bottom' )
    plt.annotate('', (blink-previous_pulse_length, 0.97), (blink, 0.97), textcoords='data', arrowprops=dict(arrowstyle='<->', mutation_aspect=1.5, mutation_scale=15))

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.annotate("D", xy=(0.02,.98), xycoords='figure fraction', fontsize='xx-large', verticalalignment='top', fontweight='bold')
plt.subplots_adjust(bottom=0.15, top=0.9, right=0.88)

plt.savefig(output_path / 'fig1D.svg')


plt.show()

