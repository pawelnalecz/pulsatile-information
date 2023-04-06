import numpy as np
from pathlib import Path
from typing import Dict

if __name__ == '__main__': import __init__

from core.experiment_manager import default_parameters, experiments, theoretical_parameters
from core.local_config import cache_directory, full_data_directory, DATA_SOURCE
from core import step_manager

from steps import (
    adjust_data,
    adjust_data_external,
    do_voting,
    extract_pulse_lengths,
    extract_slices_for_regression,
    extract_slices_for_regression_all,
    extract_slices_whole_pulses,
    extract_track_information,
    get_timeline,
    learning,
    merge_files,
    preselect_tracks,
    quantification,
    scoring,
    scoring_regression_as_yesno,
)

full_data_directory = Path(full_data_directory).absolute()

if DATA_SOURCE == 'EXTERNAL':
    adjust_data = adjust_data_external


def for_each_experiment(fun, parameters, experiment_list, unique_parameters={}, cache_directory=cache_directory, *args, **kwargs):
    prechains = {experiment: fun({
        **parameters, 
        **experiments[experiment], 
        **theoretical_parameters[experiment], 
        'pulse_length': theoretical_parameters[experiment]['minutes_per_timepoint'],
        'trim_start' : (1,0) if experiment != 'min3_mean30' else (91,0),
        'trim_end': (-1, int(np.floor(theoretical_parameters[experiment]['min'] + theoretical_parameters[experiment]['exp_mean']))),
        **(unique_parameters[experiment] if experiment in unique_parameters else {}), 
        }, cache_directory, *args, **kwargs) 
    for experiment in experiment_list}
    return prechains


def quantify_tracks(parameters=default_parameters, cache_directory=cache_directory):
    chain = step_manager.Chain(parameters, cache_directory=cache_directory, working_directory=parameters['working_directory'] or '.') \
        .step(adjust_data) \
        .step(extract_pulse_lengths) \
        .step(quantification) 
    return chain


def prepare_slices(parameters=default_parameters, cache_directory=cache_directory):
    chain = step_manager.Chain(parameters, cache_directory=cache_directory, working_directory=parameters['working_directory'] or '.') \
        .step(adjust_data) \
        .step(extract_pulse_lengths) \
        .step(quantification) \
        .step(extract_track_information) \
        .step(preselect_tracks) \
        .step(extract_slices_for_regression) 
    return chain

def deciding_if_pulse(parameters=default_parameters, cache_directory=cache_directory):
    chain = step_manager.Chain(parameters, cache_directory=cache_directory, working_directory=parameters['working_directory'] or '.') \
        .step(adjust_data) \
        .step(extract_pulse_lengths) \
        .step(quantification) \
        .step(extract_track_information) \
        .step(preselect_tracks) \
        .step(extract_slices_whole_pulses) \
        .step(learning) 
    return chain


def detecting_blink_regr(parameters=default_parameters, cache_directory=cache_directory, upsampling=False):
    chain = step_manager.Chain(parameters, cache_directory=cache_directory, working_directory=parameters['working_directory'] or '.') \
        .step(adjust_data) \
        .step(extract_pulse_lengths) \
        .step(quantification) \
        .step(extract_track_information) \
        .step(preselect_tracks) \
        .step(extract_slices_for_regression) \
        .step(learning) \
        .step(scoring_regression_as_yesno)
    return chain


def get_voting_timeline(parameters=default_parameters, parameters1=default_parameters, regular=False, onOtherDataSet=False, onGoodTracks=False, yesno=False):
    prechains = [do_the_analysis(
        parameters={**parameters, 'target_position': parameters['target_position'] + shift, 'correct_consecitive':0}, 
        parameters1={**parameters1, 'target_position': parameters1['target_position'] + shift, 'correct_consecitive':0},
        regular=regular, 
        #onOtherDataSet=onOtherDataSet, onGoodTracks=onGoodTracks, 
        yesno=yesno
        ).step(get_timeline) for shift in parameters['voting_range']]
    chain = do_the_analysis(
        parameters, 
        parameters1,
        regular=regular, 
        #onOtherDataSet=onOtherDataSet, onGoodTracks=onGoodTracks, 
        yesno=yesno
        )
    for shift,prechain in zip(parameters['voting_range'], prechains):
        chain = chain.add_file(f"binary_timeline_{shift:d}", prechain.get_file_abspath_by_type('binary_timeline'))

    return chain.step(do_voting)


def do_the_analysis(parameters=default_parameters, parameters1=default_parameters, regular=False, onOtherDataSet=False, yesno=False, upsampling=False) -> step_manager.Chain:
    if regular and not onOtherDataSet:
        chain = deciding_if_pulse(parameters).step(scoring)
    # elif regular and onOtherDataSet:
    #     chain = deciding_if_pulse_on_other_dataset(parameters1, parameters).step(scoring)

    elif not regular and not onOtherDataSet and not yesno:
        chain = detecting_blink_regr(parameters, upsampling=upsampling)
    # elif not regular and not onOtherDataSet and yesno:
    #     chain = detecting_blink_yesno(parameters)
        
    # elif not regular and onOtherDataSet and not yesno:
    #     chain = leaning_on_other_dataset(parameters1, parameters).step(scoring_regression_as_yesno)
    # elif not regular and onOtherDataSet and yesno:
    #     chain = leaning_on_other_dataset_yesno(parameters1, parameters)

    return chain

def prepare_regular_slices_from_all_experiments(parameters=default_parameters, experiment_list=experiments, cache_directory=cache_directory, working_directory=full_data_directory):

    prechains = for_each_experiment(prepare_slices, parameters, experiment_list, cache_directory=cache_directory) 

    chain = step_manager.Chain({**parameters,
        'working_directory': working_directory,
        'directory': None,
        'experiment_list': experiment_list, 
    }, cache_directory=cache_directory, working_directory=working_directory)
    
    for experiment in experiment_list:
        chain = chain.add_file('quantified_tracks_' + experiment, prechains[experiment].get_file_abspath_by_type('quantified_tracks'))
        chain = chain.add_file('extracted_slices_' + experiment, prechains[experiment].get_file_abspath_by_type('extracted_slices'))
    
    chain = chain.step(extract_slices_for_regression_all)
    return chain,prechains

def combining_results(parameters=default_parameters, prechains: Dict[str, step_manager.Chain] = {}, cache_directory=cache_directory, working_directory=full_data_directory):

    chain = combining_files({**parameters,
        'working_directory': working_directory,
        'directory': None,
    }, 
    [prechains[experiment].get_file_abspath_by_type('prediction_results') for experiment in prechains], 
    output_filetype='prediction_results', 
    labels=prechains.keys(), 
    index_name='experiment',
    cache_directory=cache_directory,
    working_directory=working_directory)
    
    return chain


def combining_files(parameters, filenames, output_filetype, labels=None, index_name='file', cache_directory=cache_directory, working_directory=full_data_directory):
    return step_manager.Chain({
        **parameters, 
        'merge_file_labels': labels, 
        'merge_file_names': filenames, 
        'merge_output_filetype': output_filetype, 
        'merge_index_name':index_name},
        cache_directory=cache_directory, working_directory=working_directory).step(merge_files)
