import pandas as pd
from typing import Literal, List

from sklearn.model_selection import train_test_split



def index_without(df: pd.DataFrame, exclude_list: List[str]):
    return [level_name for level_name in df.index.names if level_name not in exclude_list]


def split_data(slices: pd.DataFrame, train_data: pd.DataFrame, train_on: Literal['same', 'other_tracks', 'other_pulses', 'other_tracks_and_pulses'], test_set_size: int, seed: int):
    print('Splitting train and test data', end='...', flush=True)
    train_test_split_params = {
        'test_size': test_set_size,
        'shuffle': True,
    }
    if train_on == 'same':
        train_index, _ = train_test_split(train_data.index, **train_test_split_params, random_state=seed)
        _, test_index = train_test_split(slices.index, **train_test_split_params, random_state=seed)
        train_slices = train_data.reindex(train_index)
        test_slices = slices.reindex(test_index)
    elif train_on == 'other_pulses':
        train_pulses, _ = train_test_split(train_data['pulse_no'].unique(), **train_test_split_params, random_state=seed)
        _, test_pulses = train_test_split(slices['pulse_no'].unique(), **train_test_split_params, random_state=seed)
        train_slices = train_data[train_data['pulse_no'].isin(train_pulses)]
        test_slices = slices[slices['pulse_no'].isin(test_pulses)]
    elif train_on == 'other_tracks':
        train_tracks, _ = train_test_split(train_data.index.get_level_values('track_id').unique(), **train_test_split_params, random_state=seed)
        _, test_tracks = train_test_split(slices.index.get_level_values('track_id').unique(), **train_test_split_params, random_state=seed)
        train_slices = train_data[train_data.index.get_level_values('track_id').isin(train_tracks)]
        test_slices = slices[slices.index.get_level_values('track_id').isin(test_tracks)]
    elif train_on == 'other_tracks_and_pulses':
        train_tracks, _ = train_test_split(train_data.index.get_level_values('track_id').unique(), **train_test_split_params, random_state=seed)
        _, test_tracks = train_test_split(slices.index.get_level_values('track_id').unique(), **train_test_split_params, random_state=seed)
        train_pulses, _ = train_test_split(train_data['pulse_no'].unique(), **train_test_split_params, random_state=seed)
        _, test_pulses = train_test_split(slices['pulse_no'].unique(), **train_test_split_params, random_state=seed)
        train_slices = train_data[train_data.index.get_level_values('track_id').isin(train_tracks) & train_data['pulse_no'].isin(train_pulses)]
        test_slices = slices[slices.index.get_level_values('track_id').isin(test_tracks) & slices['pulse_no'].isin(test_pulses)]
    print('done', flush=True)
    return train_slices, test_slices


def first_n(iterable, n):
    i = 0
    iterator = iter(iterable)
    while i < n:
        try:
            yield next(iterator)
        except StopIteration:
            break
        i += 1


