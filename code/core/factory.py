from pathlib import Path
from typing import Callable

if __name__ == '__main__': import __init__

from core.experiment_manager import default_parameters, experiments, theoretical_parameters, trim_end
from core.local_config import cache_directory, full_data_directory, DATA_SOURCE
from core import step_manager

from steps import (
    adjust_data,
    adjust_data_external,
    do_voting_boost,
    do_voting_new,
    extract_empirical_measures_sslices,
    extract_empirical_measures_after_predictions,
    extract_pulse_lengths,
    extract_slices_binary,
    extract_slices_binary_sslices,
    extract_slices_for_regression,
    extract_slices_for_regression_all,
    extract_slices_whole_pulses,
    extract_track_information,
    get_confusion_matrix,
    learning,
    merge_files,
    MI_computation_discrete,
    MI_from_CM,
    preselect_tracks,
    quantification,
    rewrite_binary_predictions,
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
        # 'trim_start' : (1,0) if experiment != 'min3_mean30' else (91,0),
        'trim_end': trim_end(experiment),
        **(unique_parameters[experiment] if experiment in unique_parameters else {}),
        }, cache_directory=cache_directory, *args, **kwargs)
    for experiment in experiment_list}
    return prechains


def quantify_tracks(parameters=default_parameters, cache_directory=cache_directory):
    chain = step_manager.Chain(parameters, cache_directory=cache_directory, working_directory=parameters['working_directory'] or Path.cwd()) \
        .step(adjust_data) \
        .step(extract_pulse_lengths) \
        .step(quantification)
    return chain


def get_preselected_tracks(parameters=default_parameters, cache_directory=cache_directory):
    chain = quantify_tracks(parameters=parameters, cache_directory=cache_directory) \
        .step(extract_track_information) \
        .step(preselect_tracks)
    return chain


def prepare_slices_with_TAP(parameters=default_parameters, cache_directory=cache_directory):
    chain = get_preselected_tracks(parameters=parameters, cache_directory=cache_directory) \
        .step(extract_slices_for_regression)
    return chain

def prepare_slices_binary(parameters=default_parameters, cache_directory=cache_directory):
    chain = get_preselected_tracks(parameters=parameters, cache_directory=cache_directory) \
        .step(extract_slices_binary)
    return chain

def prepare_slices_binary_with_sslice(parameters=default_parameters, cache_directory=cache_directory):
    chain = get_preselected_tracks(parameters=parameters, cache_directory=cache_directory) \
        .step(extract_slices_binary_sslices)
    return chain

def prepare_slices_periodic(parameters=default_parameters, cache_directory=cache_directory, **kwargs):
    chain = get_preselected_tracks(parameters=parameters, cache_directory=cache_directory) \
        .step(extract_slices_whole_pulses)
    return chain


def prepare_slices_with_train_dataset(prepare_slices_fn: Callable[..., step_manager.Chain], parameters=default_parameters, parameters1=None, cache_directory=cache_directory):
    assert parameters1 is not None
    prechain = prepare_slices_fn(parameters=parameters1, cache_directory=cache_directory)
    chain = prepare_slices_fn(parameters=parameters, cache_directory=cache_directory) \
        .add_file('train_slices', prechain.get_file_abspath_by_type('extracted_slices'))
    return chain



def prepare_for_voting(previous_stage: Callable[..., step_manager.Chain], parameters, parameters1, chain_preprocessing=None, cache_directory=cache_directory, **kwargs) -> step_manager.Chain:
    voting_range = parameters['voting_range']
    prechains = [
        previous_stage(
            parameters={**parameters, 'target_position': parameters['target_position'] + shift},
            parameters1={**parameters1, 'target_position': parameters1['target_position'] + shift} if parameters1 is not None else None,
            **kwargs,
        ) for shift in voting_range
    ]

    required_parameters = ['voting_range']#list({'voting_range'}.union(*[prechain.step_dict[prechain.files['binary_predictions'].creator].important_parameters for prechain in prechains]))

    chain = chain_preprocessing(parameters=parameters, parameters1=parameters1, cache_directory=cache_directory) if chain_preprocessing else step_manager.Chain(parameters=parameters)

    for shift,prechain in zip(voting_range, prechains):
        chain = chain.add_file(f"binary_predictions_{shift:d}", prechain.get_file_abspath_by_type('binary_predictions'), additional_parameters=required_parameters)

    return chain



def compute_information_transmission_directly(parameters=default_parameters, parameters1=None, cache_directory=cache_directory):
    assert parameters1 is None
    assert parameters['train_on_other_experiment'] is False
    prepare_slices = prepare_slices_binary if parameters['s_slice_length'] == 1 else prepare_slices_binary_with_sslice
    chain = prepare_slices(parameters=parameters, cache_directory=cache_directory) \
        .step(MI_computation_discrete) \
        .step(extract_empirical_measures_sslices)
    return chain


def compute_information_transmission_directly_periodic(parameters=default_parameters, parameters1=None, cache_directory=cache_directory):
    assert parameters1 is None
    assert parameters['train_on_other_experiment'] is False
    chain = prepare_slices_periodic(parameters=parameters, cache_directory=cache_directory) \
        .step(MI_computation_discrete) \
        .step(extract_empirical_measures_sslices)
    return chain


def compute_information_transmission_using_reconstruction(parameters=default_parameters, parameters1=None, cache_directory=cache_directory):

    train_on_other_experiment = parameters['train_on_other_experiment']

    if train_on_other_experiment:
        assert parameters1 is not None
    else:
        assert parameters1 is None

    if train_on_other_experiment:
        chain = prepare_slices_with_train_dataset(prepare_slices_with_TAP, parameters=parameters, parameters1=parameters1, cache_directory=cache_directory)
    else:
        chain = prepare_slices_with_TAP(parameters=parameters, cache_directory=cache_directory)
        
    chain.step(learning) \
        .step(do_voting_new) \
        .step(get_confusion_matrix) \
        .step(extract_empirical_measures_after_predictions) \
        .step(MI_from_CM)
    return chain


def compute_information_transmission_using_reconstruction_periodic(parameters=default_parameters, parameters1=None,  cache_directory=cache_directory):

    train_on_other_experiment = parameters['train_on_other_experiment']

    if train_on_other_experiment:
        assert parameters1 is not None
    else:
        assert parameters1 is None


    def get_partial_predicitons(parameters, parameters1):
        if train_on_other_experiment:
            chain = prepare_slices_with_train_dataset(prepare_slices_periodic, parameters=parameters, parameters1=parameters1, cache_directory=cache_directory)
        else:
            chain = prepare_slices_periodic(parameters=parameters, cache_directory=cache_directory)
        return chain \
            .step(learning) \
            .step(rewrite_binary_predictions)
    
    chain = prepare_for_voting(get_partial_predicitons, parameters=parameters, parameters1=parameters1, chain_preprocessing=prepare_slices_periodic, cache_directory=cache_directory) \
        .step(do_voting_boost) \
        .step(get_confusion_matrix) \
        .step(extract_empirical_measures_after_predictions) \
        .step(MI_from_CM)
    return chain

def compute_information_transmission(regular: bool, learning: bool):
    if learning and not regular:
        return compute_information_transmission_using_reconstruction
    if learning and regular:
        return compute_information_transmission_using_reconstruction_periodic
    if not learning and not regular:
        return compute_information_transmission_directly
    if not learning and regular:
        return compute_information_transmission_directly_periodic
    else:
        raise ValueError(f'Arguments of the function must be boolean, but {regular=}, {learning=}')


def prepare_slices_from_all_experiments(parameters=default_parameters, experiment_list=experiments, cache_directory=cache_directory, working_directory=full_data_directory):

    prechains = for_each_experiment(prepare_slices_with_TAP, parameters, experiment_list=experiment_list, cache_directory=cache_directory) 

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


def merging_files(parameters, filenames, output_filetype, labels=None, index_name='file', cache_directory=cache_directory, working_directory=full_data_directory):
    return step_manager.Chain({
        **parameters, 
        'merge_file_labels': labels, 
        'merge_file_names': filenames, 
        'merge_output_filetype': output_filetype, 
        'merge_index_name':index_name},
        cache_directory=cache_directory, working_directory=working_directory).step(merge_files)
