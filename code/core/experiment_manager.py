# experiment_manager.py

import numpy as np
from pathlib import Path
import re

if __name__ == '__main__': import __init__

from core.local_config import full_data_directory, external_data_directory, DATA_SOURCE


# If DATA_SOURCE == 'EXTERNAL', data is imported from csvs in the form as published on Zenodo
# Scroll to the end of the file to see how the dictionaries are modified

full_data_directory = Path(full_data_directory).absolute()
external_data_directory = Path(external_data_directory).absolute()


repeated_pseudorandom_date = {1: '20210427', 2: '20210428', 3: '20210429', 4: '20210501', 5: '20210502', 6: '20210502', 7: '20210501', 8: '20210429', 9: '20210428', 10: '20210427', }

experiments = {
    'min3_mean40': {'working_directory': full_data_directory / '2019-10-03--random_expo_mean40_min3' / 'analysis', 'directory': '20191003_Random-Pulses_Seq1', 'experiment_onset': 97, 'trim_start': (1,0)},
    'min3_mean30': {'working_directory': full_data_directory / '2019-10-08--random_expo_mean30_min3' / 'analysis', 'directory': '20191008_Random-Pulses_SeqNew', 'experiment_onset': 90, 'trim_start': (91,0)},
    'min3_mean20': {'working_directory': full_data_directory / '2019-10-18--random_expo_mean20_min3' / 'analysis', 'directory': '20191018_Random-Pulses_Seq20min', 'experiment_onset': 125, 'trim_start': (1,0)},
    'min20_optmean': {'working_directory': full_data_directory / '2019-11-14--random_expo_min20_reso2' / 'analysis', 'directory': '20191114_Random-pulses_Seq20min', 'experiment_onset': 90, 'trim_start': (1,0)},
    'min20_optmeanb': {'working_directory': full_data_directory / '2019-11-09--random_expo_min20_reso2' / 'analysis', 'directory': '20191109_Random-pulses_Seq20min', 'experiment_onset': 89, 'trim_start': (1,0)},
    'min15_optmean': {'working_directory': full_data_directory / '2020-02-08--random_expo_min15_reso2' / 'analysis', 'directory': '20200209_Exp_SeqRandomTomek_min15min', 'experiment_onset': 92, 'trim_start': (1,0)},
    'min10_optmean': {'working_directory': full_data_directory / '2020-01-31--random_expo_min10_reso2' / 'analysis', 'directory': '20200131_Exp_SeqRandomTomek_min10min', 'experiment_onset': 91, 'trim_start': (1,0)},
    'min3_mean20_new': {'working_directory': full_data_directory / '2021-03-15--random_expo_mean20_min2' / 'analysis', 'directory': '20210315', 'experiment_onset': 89, 'trim_start': (1,0)},
    'min3_mean30_new': {'working_directory': full_data_directory / '2021-03-18--random_expo_mean30_min2' / 'analysis', 'directory': '20210318', 'experiment_onset': 89, 'trim_start': (1,0)},
    'min3_mean40_new': {'working_directory': full_data_directory / '2021-03-30--random_expo_mean40_min2' / 'analysis', 'directory': '20210330', 'experiment_onset': 89, 'trim_start': (1,0)},
    'min3_mean50_new': {'working_directory': full_data_directory / '2021-04-02--random_expo_mean50_min2' / 'analysis', 'directory': '20210402', 'experiment_onset': 89, 'trim_start': (1,0)},
    'min3_mean50_newb': {'working_directory': full_data_directory / '2021-04-24--random_expo_mean50_min3' / 'analysis', 'directory': '20210424', 'experiment_onset': 89, 'trim_start': (1,0)},
    'min3_mean50_new_NEW_SHUTTLETRACKER': {'working_directory': full_data_directory / '2021-04-02--random_expo_mean50_min2 - NEW' / 'analysis', 'directory': '20210402', 'experiment_onset': 89, 'trim_start': (1,0)},
    'min30_optmean_new': {'working_directory': full_data_directory / '2021-04-04--random_expo_mean13_min30' / 'analysis', 'directory': '20210404', 'experiment_onset': 89, 'trim_start': (1,0)},
    'min30_optmean_newb': {'working_directory': full_data_directory / '2021-04-21--random_expo_mean13_min30' / 'analysis', 'directory': '20210421', 'experiment_onset': 89, 'trim_start': (1,0)},
    'min30_optmean_new_NEW_SHUTTLETRACKER': {'working_directory': full_data_directory / '2021-04-04--random_expo_mean13_min30 - NEW' / 'analysis', 'directory': '20210404', 'experiment_onset': 89, 'trim_start': (1,0)},
    'min20_optmean_new': {'working_directory': full_data_directory / '2021-03-22--random_expo_mean10_min20' / 'analysis', 'directory': '20210322', 'experiment_onset': 89, 'trim_start': (1,0)},
    'min15_optmean_new': {'working_directory': full_data_directory / '2021-03-20--random_expo_mean8_min15' / 'analysis', 'directory': '20210320', 'experiment_onset': 89, 'trim_start': (1,0)},
    'min15_optmean_newb': {'working_directory': full_data_directory / '2021-04-26--random_expo_mean8_min15' / 'analysis', 'directory': '20210426', 'experiment_onset': 89, 'trim_start': (1,0)},
    'min15_optmean_new_NEW_SHUTTLETRACKER': {'working_directory': full_data_directory / '2021-03-20--random_expo_mean8_min15 - NEW' / 'analysis', 'directory': '20210320', 'experiment_onset': 89, 'trim_start': (1,0)},
    'min10_optmean_new': {'working_directory': full_data_directory / '2021-04-06--random_expo_mean6_min10' / 'analysis', 'directory': '20210406', 'experiment_onset': 89, 'trim_start': (1,0)},
    'min30_optmean_sept21': {'working_directory': full_data_directory / '2021-09-08--random_expo_mean13_min30--rep8' / 'analysis', 'directory': '20210908', 'experiment_onset': 89, 'trim_start': (1,0)},
    'min25_optmean_sept21': {'working_directory': full_data_directory / '2021-09-09--random_expo_mean11_min25--rep9' / 'analysis', 'directory': '20210909', 'experiment_onset': 89, 'trim_start': (1,0)},
    'min25_optmean_sept21b': {'working_directory': full_data_directory / '2021-09-10--random_expo_mean11_min25--rep10' / 'analysis', 'directory': '20210910', 'experiment_onset': 89, 'trim_start': (1,0)},
    'min5_optmean_sept21': {'working_directory': full_data_directory / '2021-09-11--random_expo_mean5_min5--rep11' / 'analysis', 'directory': '20210911', 'experiment_onset': 89, 'trim_start': (1,0)},
    'min5_optmean_sept21b': {'working_directory': full_data_directory / '2021-09-12--random_expo_mean5_min5--rep12' / 'analysis', 'directory': '20210912', 'experiment_onset': 89, 'trim_start': (1,0)},
    **{ f'pseudorandom_pos{pos:02d}_period{period:d}_new': {'working_directory': full_data_directory / '2021-04-27--pseudorandom' / 'analysis', 'directory': f'{repeated_pseudorandom_date[pos]}--pos_{1 if pos <= 5 else 2:d}--period_{period:02d}', 'experiment_onset': 29 if period == 10 else 60, 'trim_start': (0,0)}
        for period in [3, 10, 15] for pos in range(1,11) },
    **{ f'pseudorandom_pos{pos+1:02d}_period5': {'working_directory': full_data_directory / '2021-04-07--pseudorandom' / 'analysis', 'directory': f'20210407_pos{pos:d}_part1', 'experiment_onset': 29, 'trim_start': (0,0)}
        for pos in range(1,9) },
    **{ f'pseudorandom_pos{pos+1:02d}_period7': {'working_directory': full_data_directory / '2021-04-07--pseudorandom' / 'analysis', 'directory': f'20210407_pos{pos:d}_part2', 'experiment_onset': 39, 'trim_start': (0,0)}
        for pos in range(1,9) },
    **{ f'pseudorandom_pos{pos:02d}_period10': {'working_directory': full_data_directory / '2019-09-03--pseudorandom' / 'analysis', 'directory': f'Pos{pos:02d}--t_1395_1675--period_10', 'experiment_onset': 30, 'trim_start': (0,0)}
        for pos in range(1,11) }, 
    **{ f'pseudorandom_pos{pos:02d}_period15': {'working_directory': full_data_directory / '2019-09-03--pseudorandom' / 'analysis', 'directory': f'Pos{pos:02d}--t_1050_1395--period_15', 'experiment_onset': 30, 'trim_start': (0,0)}
        for pos in range(1,11) }, 
    **{ f'pseudorandom_pos{pos:02d}_period20': {'working_directory': full_data_directory / '2019-09-03--pseudorandom' / 'analysis', 'directory': f'Pos{pos:02d}--t_0610_1050--period_20', 'experiment_onset': 30, 'trim_start': (0,0)}
        for pos in range(1,11) }, 
    **{ f'pseudorandom_pos{pos:02d}_period30': {'working_directory': full_data_directory / '2019-09-03--pseudorandom' / 'analysis', 'directory': f'Pos{pos:02d}--t_0000_0610--period_30', 'experiment_onset': 10, 'trim_start': (0,0)}
        for pos in range(1,11) }, 
    # 'pseudorandom_pos1_repeated':  {'working_directory': full_data_directory / '2021-04-27--pseudorandom--repl1_2pos' / 'analysis', 'directory': '20210427A'}, 
    # 'pseudorandom_pos2_repeated':  {'working_directory': full_data_directory / '2021-04-28--pseudorandom--repl2_2pos' / 'analysis', 'directory': '20210428A'}, 
    # 'pseudorandom_pos3_repeated':  {'working_directory': full_data_directory / '2021-04-29--pseudorandom--repl3_2pos' / 'analysis', 'directory': '20210429A'}, 
    # 'pseudorandom_pos4_repeated':  {'working_directory': full_data_directory / '2021-05-01--pseudorandom--repl4_2pos' / 'analysis', 'directory': '20210501A'}, 
    # 'pseudorandom_pos5_repeated':  {'working_directory': full_data_directory / '2021-05-02--pseudorandom--repl5_2pos' / 'analysis', 'directory': '20210502A'}, 
    # 'pseudorandom_pos6_repeated':  {'working_directory': full_data_directory / '2021-05-02--pseudorandom--repl5_2pos' / 'analysis', 'directory': '20210502A'}, 
    # 'pseudorandom_pos7_repeated':  {'working_directory': full_data_directory / '2021-05-01--pseudorandom--repl4_2pos' / 'analysis', 'directory': '20210501A'}, 
    # 'pseudorandom_pos8_repeated':  {'working_directory': full_data_directory / '2021-04-29--pseudorandom--repl3_2pos' / 'analysis', 'directory': '20210429A'}, 
    # 'pseudorandom_pos9_repeated':  {'working_directory': full_data_directory / '2021-04-28--pseudorandom--repl2_2pos' / 'analysis', 'directory': '20210428A'}, 
    # 'pseudorandom_pos10_repeated': {'working_directory': full_data_directory / '2021-04-27--pseudorandom--repl1_2pos' / 'analysis', 'directory': '20210427A'}, 
        
}

theoretical_parameters = {
    'min3_mean20': {'min': 2, 'exp_mean': 25, 'minutes_per_timepoint': 1},#20
    'min3_mean20_new': {'min': 2, 'exp_mean': 20, 'minutes_per_timepoint': 1},
    'min3_mean30': {'min': 2, 'exp_mean': 35, 'minutes_per_timepoint': 1},#30
    'min3_mean30_new': {'min': 2, 'exp_mean': 30, 'minutes_per_timepoint': 1},
    'min3_mean40': {'min': 2, 'exp_mean': 50, 'minutes_per_timepoint': 1},#40
    'min3_mean40_new': {'min': 2, 'exp_mean': 40, 'minutes_per_timepoint': 1},
    # 'min3_mean50_new': {'min': 2, 'exp_mean': 50, 'minutes_per_timepoint': 1},
    # 'min3_mean50_newb': {'min': 2, 'exp_mean': 50, 'minutes_per_timepoint': 1},
    # 'min3_mean50_new_NEW_SHUTTLETRACKER': {'min': 2, 'exp_mean': 50, 'minutes_per_timepoint': 1},
    'min5_optmean_sept21': {'min': 5, 'exp_mean': 4.506323230388831, 'minutes_per_timepoint': 1}, 
    'min5_optmean_sept21b': {'min': 5, 'exp_mean': 4.506323230388831, 'minutes_per_timepoint': 1}, 
    'min10_optmean': {'min': 10, 'exp_mean': 6.426632937053691, 'minutes_per_timepoint': 1}, # 1/0.155602471208=6.4266331520099085
    'min10_optmean_new': {'min': 10, 'exp_mean': 6.426632937053691, 'minutes_per_timepoint': 1},
    'min15_optmean':      {'min': 15, 'exp_mean': 8.141126142481896, 'minutes_per_timepoint': 1}, # 8.141126058563819 #1/0.122833130553
    'min15_optmean_new':  {'min': 15, 'exp_mean': 8.141126142481896, 'minutes_per_timepoint': 1},
    'min15_optmean_newb': {'min': 15, 'exp_mean': 8.141126142481896, 'minutes_per_timepoint': 1},
    'min15_optmean_new_NEW_SHUTTLETRACKER': {'min': 16, 'exp_mean': 8.141126142481896, 'minutes_per_timepoint': 1},
    'min20_optmean': {'min': 20, 'exp_mean': 9.736304000718421, 'minutes_per_timepoint': 1}, # 1/0.102708377816=9.73630409966634 # used to be 2/0.155602471208 = 12.853266304019817
    'min20_optmeanb': {'min': 20, 'exp_mean': 9.736304000718421, 'minutes_per_timepoint': 1}, #1/0.102708377816= 9.73630409966634
    'min20_optmean_new': {'min': 20, 'exp_mean': 9.736304000718421, 'minutes_per_timepoint': 1},
    'min25_optmean_sept21': {'min': 25, 'exp_mean': 11.249944864977316, 'minutes_per_timepoint': 1},
    'min25_optmean_sept21b': {'min': 25, 'exp_mean': 11.249944864977316, 'minutes_per_timepoint': 1},
    'min30_optmean_new': {'min': 30, 'exp_mean': 12.702805026377764, 'minutes_per_timepoint': 1},
    'min30_optmean_newb': {'min': 30, 'exp_mean': 12.702805026377764, 'minutes_per_timepoint': 1},
    'min30_optmean_sept21': {'min': 30, 'exp_mean': 12.702805026377764, 'minutes_per_timepoint': 1},
    'min30_optmean_new_NEW_SHUTTLETRACKER': {'min': 30, 'exp_mean': 12.702805026377764, 'minutes_per_timepoint': 1},
    **{ f'pseudorandom_pos{pos:02d}_period{period:d}': {'min' : 0, 'exp_mean': 2, 'minutes_per_timepoint': period} for pos in range(1,11) for period in[5, 7, 10, 15, 20, 30]},
    **{ f'pseudorandom_pos{pos:02d}_period{period:d}_new': {'min' : 0, 'exp_mean': 2, 'minutes_per_timepoint': period} for pos in range(1,11) for period in[3, 10, 15]},
}


def trim_end(experiment):
    return (-1, int(np.floor(theoretical_parameters[experiment]['min'] + theoretical_parameters[experiment]['exp_mean'])))
                


def map_to_official_naming(experiment: str):

    if experiment == 'min3_mean20':
        return 'min3_mean25'
    if experiment == 'min3_mean30':
        return 'min3_mean35'
    if experiment == 'min3_mean40':
        return 'min3_mean50'
    if experiment == 'min3_mean50_new':
        return 'min3_mean50_rep2'
    if experiment == 'min3_mean50_newb':
        return 'min3_mean50_rep3'
        

    if 'pseudorandom' in experiment:
        pos_position = experiment.find('pos')
        pos = int(experiment[pos_position+3:pos_position+5])
        if experiment[-2:] in ('20', '30'):
            experiment = experiment.replace(experiment[pos_position:pos_position+3], f"rep1_pos")
        elif experiment[-2:] in ('10', '15'):
            experiment = experiment.replace(experiment[pos_position:pos_position+3], f"rep0_pos")
        else:
            hasOnlyFour = experiment[-1:] in ['5', '7']
            experiment = experiment.replace(experiment[pos_position:pos_position+5], f"rep{(pos if pos <=5 else 11 - pos)-hasOnlyFour:d}_pos{1+1*(pos>5):02d}")
    elif 'optmean' in experiment:
        rep_no = 1 if experiment in ['min5_optmean_sept21', 'min10_optmean', 'min15_optmean', 'min20_optmean', 'min25_optmean_sept21', 'min30_optmean_newb'] else \
                 2 if experiment in ['min5_optmean_sept21b', 'min10_optmean_new', 'min15_optmean_newb', 'min20_optmeanb', 'min25_optmean_sept21b', 'min30_optmean_new'] \
                 else 3
        experiment = experiment +f"_rep{rep_no}"
    
    experiment = experiment.replace('_sept21b', '')
    experiment = experiment.replace('_sept21', '')
    experiment = experiment.replace('_newb', '')
    experiment = experiment.replace('_new', '')
    experiment = experiment.replace('optmeanb', 'optmean')
    experiment = experiment.replace('pseudorandom', 'binary')

    return experiment
        
internal_to_official_naming = {experiment: map_to_official_naming(experiment) for experiment in experiments.keys()}
official_to_internal_naming = {val: key for key,val in internal_to_official_naming.items()}

assert len(internal_to_official_naming) == len(official_to_internal_naming), "Internal to official naming is not unique"

def get_official_directory(experiment):
    official_name = internal_to_official_naming[experiment]
    experiment_type = 'binary_encoding' if 'pseudorandom' in experiment else 'interval_encoding_with_minimal_gap' if 'optmean' in experiment else 'interval_encoding'
    return Path(external_data_directory) / experiment_type / official_name




chosen_experiments_pseudorandom = [
    *[f'pseudorandom_pos{pos:02d}_period{period:d}' for period in [20, 30] for pos in range(1,11) ],
    *[f'pseudorandom_pos{pos:02d}_period{period:d}' for period in [5, 7] for pos in range(2,10) ],
    *[f'pseudorandom_pos{pos:02d}_period{period:d}_new' for period in [3, 10, 15] for pos in range(1,11)],
]

chosen_experiments_interval = [
    'min3_mean20',
    'min3_mean30',
    'min3_mean40',
    'min3_mean20_new',
    'min3_mean30_new',
    'min3_mean40_new',
]

chosen_experiments_interval_with_gap = [
    # 'min5_optmean_sept21b',
    'min10_optmean',
    'min10_optmean_new',
    'min15_optmean',
    'min15_optmean_newb',
    'min20_optmean',
    'min20_optmeanb',
    # 'min20_optmean_new',
    'min25_optmean_sept21',
    'min25_optmean_sept21b',
    'min30_optmean_newb',
]

best_experiments = [
    'min3_mean30',
    'min3_mean30_new',
    'min3_mean40_new',

    'min15_optmean',
    'min15_optmean_newb',
    'min20_optmean',
    'min20_optmeanb',
    # 'min20_optmean_new',

    *[ f'pseudorandom_pos{pos:02d}_period{period:d}' for pos in range(1,11) for period in [5, 7]],
    # *[ f'pseudorandom_pos{pos:02d}_period{period:d}_new' for pos in range(1,11) for period in [3]],
    # 3,
    5,
    7,

]

best_experiments_new = [
    'min3_mean20', 
    'min3_mean30', 
    'min3_mean20_new', 
    'min3_mean30_new',  

    'min10_optmean',
    'min10_optmean_new', 
    'min15_optmean', 
    'min15_optmean_newb', 
    'min20_optmean', 
    'min20_optmeanb',
    # 'min20_optmean_new',

    *[ f'pseudorandom_pos{pos:02d}_period{period:d}_new' for pos in range(1,11) for period in [3]],
    3,

]

default_parameters = {
    'working_directory': experiments['min20_optmean']['working_directory'], # working dir should be the directory where the data is located
    'directory': experiments['min20_optmean']['directory'],                 # directory where original images were stored; serves as experiment id
    'theoretical_parameters': theoretical_parameters['min20_optmean'],      # assumed theoretical parameters of the inter-spike distribution
    'n_tracks': 'None',                                                     # number of tracks, selected based on the quality of tracking. Should match the pickle name. 'None' means take all. Note that this default is changed at the end of this file if DATA_SOURCE == 'EXTERNAL'
    'vivid_track_criteria': [                                               # criteria for track preselection. (field, method, comparison_method, threshold)
        ('', 'index', 'lt', 500),
        ('std_dQ2', 'rank', 'gt', 0.2),
    ],
    'take_tracks':  'preselected',                                          # list of indices of tracks to be used in the analysis. Use 'full' to take all tracks and 'preselected' to take all 'vivid' tracks determined in the preselection step
    'trim_start': (1,0),                                                    # (pulse_no, offset). Trim the timeline to start {offset} timepoints after the {pulse_no}th pulse. 
    'trim_breaks_longer_than': 240,                                         # eliminate breaks in the stimulation longer than the given number of timepoints. Used to exclude the 4h break artificially introduced in the middle of (some of) the experiment to allow for cell regeneration
    'fields_for_learning': ['dQ3backw'],                                    # quantification methods used for reconstruction/entropy estimation. Q1=raw signal, Q2=normalized with image mean, Q3=Q2 normalized with 120min history around the timepoint, Q3backw=Q2 normalized with 120min backward history, Q4=Q2 normailzed with whole history, Q5=gaussian smoothing of Q3backw. Add prefix d for derivative
    'slice_length': 5,                                                      # number of timepoints used for reconstruction/entropy estimation. Note that for derivative, slice_length=k utilizes k finite differences, i.e., (k+1) timepoints in original time series
    'target_position': 4,                                                   # time after pulse (TAP), i.e. offset from a time point potentially containing a pulse to the latest point in the slice associated it and u
    'nearest_neighbor_k': 20,                                               # parameter k of the kNN classifier
    'n_iters': 10,                                                          # number of train/test partitionings
    'test_set_size': 0.5,                                                   # fraction of slices in the test set
    'train_on': 'other_tracks',                                             # whether to prevent using slices from the same track and/or the same timepoint in the train and test sets simultaneously
    'train_on_other_experiment': False,                                     # whether to train the classisier based on data from a complementary experiment (reconstruction-based approach only)
    'pulse_window_matching_shift': 3,                                       # minimal TAP used for labeling; time points with TAP < pulse_window_matching_shift are labeled with respect to the previous pulse
    'voting_range': [-1, 0, 1],                                             # TAP offets with respect to target_position to use for voting; used only in binary encoding
    'voting_threshold': 2,                                                  # number of positive votes to classify a timepoint as containing a pulse
    'correct_consecutive': 2,                                               # force {correct_consecutive} zeros after each 1 in the reconstruction
    'entropy_estimation_method': 'naive_with_MM_correction',                # method of entropy estimation in the reconstruction-free approach
    'fields_reduced_with_confusion_matrix': ['track_id', 'time_point'],     # dimensions to be reduced in confusion matrix computations
    's_slice_length': 1,                                                    # length of chunk in the S (signal) sequence used for entropy computation
    'r_slice_length': 3,                                                    # length of chunk in the R (reconstruction) sequence used for entropy computation
 


}


if DATA_SOURCE == 'EXTERNAL':
    for experiment_name, experiment in experiments.items():
        experiment['directory'] = internal_to_official_naming[experiment_name]
        experiment['working_directory'] = get_official_directory(experiment_name).parent
    default_parameters['n_tracks'] = '' # This is because data on Zenodo have no appendix. Effectively, '' means 500
    

def get_complementary_experiment(experiment):
    pos_match = re.search('pos([0-9]+)_', experiment)
    assert pos_match is not None, f'Position in experiment {experiment} cannot be established'
    pos = int(pos_match.group(1))
    complementary_pos_text = f'pos{11-pos:02d}_'
    print(experiment, pos_match, pos, complementary_pos_text, pos_match.group(0), experiment.replace(pos_match.group(0), complementary_pos_text))
    return experiment.replace(pos_match.group(0), complementary_pos_text)

