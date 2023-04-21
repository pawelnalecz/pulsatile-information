from matplotlib import pyplot as plt
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from figures.local_config import figure_output_path
from figures.figure_settings import *
from integrity import check_and_fetch

check_and_fetch.check_and_fetch_necessary()

output_path = Path(figure_output_path).absolute() / "fig1/automatic/"
output_path.mkdir(parents=True, exist_ok=True)



plt.figure(figsize=(7,4))
for subplot_no, (title, description, blinks, the_blink, regular) in enumerate(( 
    ('Binary\nencoding', '$\\tau_{clock}$', pd.Series([0, 5, 20, 30, 35, 45, 60, 65]), 2, 5),
    ('Interval\nencoding', '$\sim Geom(1/\\tau_{geom})$', pd.Series([0, 6, 27, 30, 48, 56, 60, 69]), 2, False),
    ('Interval\nencoding\n with gap', '$\sim \\tau_{gap} + Geom(1/\\tau_{geom})$', pd.Series([2, 19, 35, 49, 68]), 2, False),
)):
    plt.subplot(3, 1, subplot_no+1)

    xlim=(-3,70)
    previous_pulse_lengths = blinks.diff()

    potential_blinks = blinks if not regular else [x  for x in range(*xlim) if not x % regular]
    minutes_per_timepoint=regular

    plt.vlines(blinks, 0.02, 1, lw=5)
    for blink,previous_pulse_length in zip(potential_blinks, previous_pulse_lengths if not regular else [np.nan] + [minutes_per_timepoint]*(len(potential_blinks)-1)):
        if blink < xlim[0] or blink >= xlim[1]:
            continue
        plt.annotate(f'{previous_pulse_length:.0f}', (blink-previous_pulse_length/2, 0.0 - 0.08), fontweight='bold', horizontalalignment='center', verticalalignment='top', fontsize='large')
        # plt.hlines([-0.75], blink-previous_pulse_length, blink, 'k')
        plt.plot(potential_blinks, [0.0-0.13]*len(potential_blinks), c='k', marker='|', lw=0.3, ms=10, ls='none')
    if not regular:
        plt.annotate(description, (blinks[the_blink] - previous_pulse_lengths[the_blink]/2 - 4, 0.65 + 0.06), fontweight='bold', horizontalalignment='left', verticalalignment='bottom', fontsize='large', rotation=15)
        plt.annotate('', (blinks[the_blink]-previous_pulse_lengths[the_blink], 0.65 +0.02), (blinks[the_blink], 0.65  +0.02), textcoords='data', arrowprops=dict(arrowstyle='<->', mutation_aspect=1.5, mutation_scale=15, shrinkA=0, shrinkB=0))#shape='full', length_includes_head=True, width=0.00001, head_width=0.05, head_length=0, facecolor='k')
    if regular:
        print(blinks, potential_blinks)
        for blink in potential_blinks:
            plt.annotate('1' if blink in blinks.tolist() else '0', (blink, 1.1), horizontalalignment='center', fontsize='large')
        for the_blink in range(2, 3):
            plt.annotate(description, (potential_blinks[the_blink]-minutes_per_timepoint/2-2, 1.2 + 0.06), fontweight='bold', horizontalalignment='left', verticalalignment='bottom', fontsize='large', rotation=15, textcoords='offset points', xytext=(0,4))
            plt.annotate('', (potential_blinks[the_blink]-minutes_per_timepoint, 1.2 +0.02), (potential_blinks[the_blink], 1.2  +0.02), textcoords='data', arrowprops=dict(arrowstyle='<->', mutation_aspect=1.5, mutation_scale=15, shrinkA=1.2, shrinkB=1.2))#shape='full', length_includes_head=True, width=0.00001, head_width=0.05, head_length=0, facecolor='k')
        
    plt.hlines([0.02], *xlim, color='k')


    plt.ylim(-0.15,1.28)
    plt.xlim(xlim)
    plt.yticks([])
    if subplot_no == 2: 
        plt.xlabel('Time $\\longrightarrow$', fontsize='large', labelpad=10)
    plt.xticks([])
    plt.ylabel(title, fontsize='large')
    plt.gca().spines['bottom'].set(visible=False)
    plt.gca().spines['top'].set(visible=False)
    plt.gca().spines['left'].set(visible=False)
    plt.gca().spines['right'].set(visible=False)
# plt.subplots_adjust(top=0.74)
plt.subplots_adjust(hspace=1, top=0.85)

plt.annotate("A", xy=(0.02,.98), xycoords='figure fraction', fontsize='xx-large', verticalalignment='top', fontweight='bold')
plt.savefig(output_path / 'fig1B.svg')


plt.show()

