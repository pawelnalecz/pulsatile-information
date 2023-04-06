# -*- coding: utf-8 -*-

"""This module reads in quantification data and tracks generated
   by ShuttleTracker, a program for semi-automated cell tracking.
   Visit http://pmbm.ippt.pan.pl/software/shuttletracker to learn
   more on this software. The code is licenced under GNU GPL v3.


   Example 1
   ---------
   This module can be used as follows:

       from shuttletracker import load_tracks, read_time_interval

       my_dir = 'my_dir_with_quant_files_and_tracks_folder_and_shuttletracker_metadata_txt'
       delta_t = read_time_interval(my_dir)
       for track_name, q, is_revised in load_tracks(my_dir, ntracks=50):
           if is_revised:
               x = q['time_point_index']*delta_t
               y = q['nuc_Protein1_intensity_mean']/q['per_Protein1_intensity_mean']
               # Now, plot y versus x using your favorite plotting package.


    Example 2
    ---------
    Warn about the most outstanding errors in segmentation or tracking:

        import shuttletracker

        my_dir = 'my_dir_with_quant_files_and_tracks_folder_and_shuttletracker_metadata_txt'
        delta_t = read_time_interval(my_dir)
        qtracks = load_tracks(my_dir):

        shuttletracker.inspect(qtracks)


    Example 3
    ---------
    Assess tracks one-by-one:

        import shuttletracker

        my_dir = 'my_dir_with_quant_files_and_tracks_folder_and_hints_for_shuttletracker_txt'
        delta_t = read_time_interval(my_dir)
        qtracks = load_tracks(my_dir):

        inspection = shuttletracker.inspect(qtracks, print_results=False)
        shuttletracker.print_inspection_per_track(qtracks, inspection)
"""

import os
import glob
import re
import multiprocessing as mp
import pandas as pd
import numpy as np
from timeit import default_timer as timer


QUANT_KINDS = 'nuc per img reg'.split()
QUANTS = None

TIMEPOINT_RE = re.compile('_?[TtZz]([0-9]+)')
CHANNEL_RE = re.compile('_?[Cc][Hh]?([0-9]+)')


def _annotate_with_quants(track_with_revision_status):
    """This is an auxiliary function called by load_tracks()."""

    assert QUANTS

    track_name, track, is_track_revised = track_with_revision_status

    to_concat = []
    for _, time_point_entry in track.iterrows():
        time_point_index, nucleus_id = time_point_entry

        q_nuc = QUANTS['nuc'][time_point_index]
        nucleus_quants = q_nuc[q_nuc['nucleus_id'] == nucleus_id].copy()
        nucleus_quants.rename(columns=lambda cn: 'nuc_'+cn, inplace=True)

        merged_quants = nucleus_quants
        merged_quants.insert(0, 'time_point_index', time_point_index)

        if 'per' in QUANTS.keys() and QUANTS['per']:
            q_pnuc = QUANTS['per'][time_point_index]
            perinucleus_quants = q_pnuc[q_pnuc['corresp_nucleus_id'] == nucleus_id].copy()
            perinucleus_quants.rename(columns=lambda cn: 'per_'+cn, inplace=True)
            merged_quants = pd.merge(merged_quants, perinucleus_quants,
                                     left_on='nuc_nucleus_id', right_on='per_corresp_nucleus_id')

        merged_quants.reset_index(inplace=True, drop=True)

        if 'img' in QUANTS.keys() and QUANTS['img']:
            image_quants = QUANTS['img'][time_point_index].copy()
            image_quants.rename(columns=lambda cn: 'img_'+cn, inplace=True)
           #merged_quants = pd.concat([merged_quants, image_quants], axis='columns')
           # Numpy's hstack() is faster than `pd.concat([merged_qs, image_qs], axis='columns')`:
        if not to_concat:
            col_types = merged_quants.dtypes.append(image_quants.dtypes)
            col_names = merged_quants.columns.append(image_quants.columns)

        merged_quants_values = np.hstack([merged_quants.values, image_quants.values])
        to_concat.append(merged_quants_values)

    #track_with_quants = pd.concat(to_concat, axis='index')
    ## Numpy's concatenate() is faster than `pd.concat(to_concat, axis='index')`:
    track_quants = pd.DataFrame(np.concatenate([v for v in to_concat]), columns=col_names)
    track_quants = track_quants.astype(col_types)

    track_quants.set_index('time_point_index', inplace=True)

    return (track_name, track_quants, is_track_revised)



def _timepoint_then_channel(filename):

    timepoint_match = re.search(TIMEPOINT_RE, filename)
    if not timepoint_match is None:
        timepoint_index = int(timepoint_match.group(1))
    else:
        timepoint_index = -1

    channel_match = re.search(TIMEPOINT_RE, filename)
    if not channel_match is None:
        channel_index = int(channel_match.group(1))
    else:
        channel_index = -1

    order = 0.
    if timepoint_index >= 0:
        order = float(timepoint_index)
    if channel_index >= 0:
        order += float(channel_index)/1000.

    return order


def _find_quantification_files_and_tracks(working_dir):
    """This is an auxiliary function called by load_tracks()."""

    _quant_file_suffix = '_quant.csv'
    _tracks_file_name = 'tracks.csv'
    _tracks_meta_file_name = 'tracks_meta.csv'


    # -- initial checks

    if not os.path.exists(working_dir):
        print('Error: directory {} does not exist.'.format(working_dir))
        return (None, None, None)

    if not os.path.isdir(working_dir):
        print('Error: path {} is not a directory.'.format(working_dir))
        return (None, None, None)


    # -- find files with quantifications

    quant_files_paths = dict()
    for kind in QUANT_KINDS:
        file_name_pattern = working_dir + os.sep + '*' + kind + _quant_file_suffix
        quant_files_paths[kind] = sorted(glob.glob(file_name_pattern),
                                         key=_timepoint_then_channel)

    quant_files_counts = {kind:len(quant_files_paths[kind]) for kind in QUANT_KINDS}
    quant_files_counts_non_zero = {v for v in quant_files_counts.values() if v != 0}

    if not quant_files_counts_non_zero:
        print('Error: quantification files not found.')
        return (None, None, None)

    if len(quant_files_counts_non_zero) != 1:
        print('Error: numbers of quantification files do not match.')
        print(quant_files_counts)
        return (None, None, None)

    quant_files_count = list(quant_files_counts_non_zero)[0]
    quant_kinds_present = [kind for kind, count in quant_files_counts.items()
                           if count == quant_files_count]

    if not 'nuc' in quant_kinds_present:
        print("Error: no nuclear ('nuc') quantifications found.")
        return (None, None, None)

    print('Found {} quantification file sets ({}).'
          .format(quant_files_count, ','.join(quant_kinds_present).replace(',', ', ')))


    # -- find both tracks-related files

    tracks_file_path = working_dir + os.sep + _tracks_file_name
    tracks_meta_file_path = working_dir + os.sep + _tracks_meta_file_name

    if not os.path.exists(tracks_file_path):
        print('Error: tracks file {} does not exist.'.format(tracks_file_path))
        return (None, None, None)

    if not os.path.exists(tracks_meta_file_path):
        print('Error: tracks meta file {} does not exist.'.format(tracks_file_path))
        return (None, None, None)

    n_tracks = len(set(pd.read_csv(tracks_file_path, error_bad_lines=True)['track_index'].values))
    print('Found {} tracks.'.format(n_tracks))

    return (quant_files_paths, tracks_file_path, tracks_meta_file_path)



def load_tracks(working_dir, ntracks=None, exclude_tracks_with_incomplete_nuclei=False,
                only_revised=False):

    """Read in all available quantification files and tracks.

     Parameters
     ----------
        working_dir : string
            path to a directory that contains quantification DAT files
            generated by ShuttleTracker as well as directory 'tracks'

        exclude_tracks_with_incomplete_nuclei : bool

        only_revised : bool

    Returns
    -------
        list of pd.core.frame.DataFrame
            each data frame in the list is a track with nuclear and
            (if possible) perinuclear quantifications
    """

    quant_files_paths, tracks_file_path, tracks_meta_file_path = \
        _find_quantification_files_and_tracks(working_dir)

    if quant_files_paths is None \
            or tracks_file_path is None \
            or tracks_meta_file_path is None:
        print('Error: cannot load tracks.')
        return []


    # -- load quantifications and tracks

    global QUANTS
    QUANTS = dict()

    print('Reading quantification files...', end='', flush=True)
    t_begin = timer()
    for kind in QUANT_KINDS:
        QUANTS[kind] = []
        for quant_file_path in quant_files_paths[kind]:
            dframe = pd.read_csv(quant_file_path, error_bad_lines=True)
            if dframe.size == 0:
                print('Warning: no data in file "{}".'.format(quant_file_path))
            QUANTS[kind].append(dframe)

    tracks = []
    tracks_data = pd.read_csv(tracks_file_path, error_bad_lines=True)
    tracks_metadata = pd.read_csv(tracks_meta_file_path, error_bad_lines=True)
    tracks_data_indices = sorted(list(set(tracks_data['track_index'].values)))
    tracks_metadata_indices = sorted(list(set(tracks_metadata['track_index'].values)))
    assert tracks_data_indices == tracks_metadata_indices
    for track_i in tracks_data_indices:
        track_name = str(track_i + 1)
        track_metadata = tracks_metadata[tracks_metadata['track_index'] == track_i]
        is_track_revised = bool(track_metadata['track_revision_status'].values[0])
        if not only_revised or only_revised and is_track_revised:
            track_data = tracks_data[tracks_data['track_index'] == track_i]
            tracks.append((track_name,
                           track_data[['time_point_index', 'nucleus_index']],
                           is_track_revised))
        if ntracks and track_i + 1 >= ntracks:
            break
    duration = timer() - t_begin
    print(' done ({:d}m{:d}s).'.format(int(duration//60), int(duration%60)), flush=True)
    if not tracks:
        return []

    # -- merge quantifications and tracks

    printed_ntracks = ntracks if ntracks is not None else len(tracks)
    t_begin = timer()
    print('Merging {:d} tracks with quantifications...'.format(printed_ntracks),
          end='', flush=True)
    with mp.Pool() as pool:
        tracks_with_quants = list(pool.map(_annotate_with_quants, tracks))
    duration = timer() - t_begin
    print(' done ({:d}m{:d}s).'.format(int(duration//60), int(duration%60)), flush=True)

    # -- if requested, excluded tracks with nuclei marked as incomplete

    if exclude_tracks_with_incomplete_nuclei:
        tracks_to_be_excluded = []
        for tracki, (_, qtrack, __) in enumerate(tracks_with_quants):
            any_nuc_incomplete = 0 in qtrack['nuc_is_complete'].values
            if any_nuc_incomplete:
                tracks_to_be_excluded += [tracki]
        for tracki in sorted(tracks_to_be_excluded, reverse=True):
            del tracks_with_quants[tracki]
        print('Excluding {:d} tracks with incomplete nuclei... done.'
              .format(len(tracks_to_be_excluded)), end='', flush=True)

    return tracks_with_quants



def read_time_interval(working_dir):
    """Extracts time interval from the 'shuttletracker_metadata.txt' file."""

    _hints_filename = 'shuttletracker_metadata.txt'
    _time_interval_keyword = 'time_interval'
    _default_time_interval = 1.

    hints_file_path = working_dir + os.sep + _hints_filename
    if os.path.exists(hints_file_path):
        with open(hints_file_path) as hints_file:
            for line in hints_file:
                if _time_interval_keyword in line:
                    assert line.split()[0] == _time_interval_keyword
                    return float(line.split()[1])
    else:
        print('Warning: could not find "{}".'.format(hints_file_path))

    print('Warning: assuming default time interval {}.'.format(_default_time_interval))
    return _default_time_interval



def _calculate_displacements_squared_descending(qtracks):
    """This is an auxiliary function called by _check_displacements_sanity()
       and suggest_merging()."""

    displacements = []
    for track_name, qtrack, _ in qtracks:
        prev_xy = (None, None)
        for __, row in qtrack.iterrows():
            curr_xy = tuple(row[['nuc_center_x', 'nuc_center_y']])
            curr_time_frame = int(row['time_point_index'])
            if all(prev_xy):
                d_sq = (curr_xy[0] - prev_xy[0])**2 + (curr_xy[1] - prev_xy[1])**2
                displacements += [((track_name, curr_time_frame), d_sq)]
            prev_xy = curr_xy

    return sorted(displacements, key=lambda z: z[-1], reverse=True)



def check_displacements_sanity(qtracks, print_results, print_most_displaced_n):
    """This is an auxiliary function called by inspect()."""

    displacements = _calculate_displacements_squared_descending(qtracks)

    assert displacements
    assert displacements == sorted(displacements, key=lambda z: z[-1], reverse=True)
    mid_position = len(displacements)//2
    median_d_sq = displacements[mid_position][-1]

    returned_displacements = []
    if print_results:
        print('Largest nuclear displacements [showing largest {}]:'.format(print_most_displaced_n))
    for i, ((track_name, time_frame), d_sq) in enumerate(displacements):
        if i >= print_most_displaced_n:
            break
        d_in_median_d = d_sq**0.5/median_d_sq**0.5
        returned_displacements += [(track_name, time_frame, d_in_median_d)] # track No. is 1-based
        if print_results:
            print('  track {:3s} [@ time-point {:3d}]: '
                  'nucleus displacement of {:.1f}x median displacement'
                  .format(track_name, time_frame, d_in_median_d))
    if print_results:
        print()

    return returned_displacements



def check_nuclear_area_consistency(qtracks,
                                   nuclear_area_fraction_drop_threshold,
                                   nuclear_area_fraction_surge_threshold,
                                   print_results):
    """Report when a nuclear area drop is >= threshold.
    This is an auxiliary function called by inspect()."""

    drops, surges = [], []
    for track_name, qtrack, _ in qtracks:
        prev_area = None
        for __, row in qtrack.iterrows():
            curr_area = row['nuc_area']
            curr_time_frame = int(row['time_point_index'])
            if not prev_area:
                prev_area = curr_area
                continue
            else:
                area_rel_change = (curr_area - prev_area)/prev_area
                if area_rel_change <= -nuclear_area_fraction_drop_threshold:
                    drops += [((track_name, curr_time_frame), area_rel_change)]
                if area_rel_change >= nuclear_area_fraction_surge_threshold:
                    surges += [((track_name, curr_time_frame), area_rel_change)]
                prev_area = curr_area

    drops = sorted(drops, key=lambda z: z[-1])
    surges = sorted(surges, key=lambda z: z[-1], reverse=True)

    if print_results:
        print('Nuclear area drops [larger than {:.1f}%]:'
              .format(nuclear_area_fraction_drop_threshold*100))
        for (track_name, time_frame), area_rel_change in drops:
            print('  track {:3s} [@ time-point {:3d}]: nucleus area {:.1f}%'
                  .format(track_name, time_frame, 100*area_rel_change))
        print()
        print('Nuclear area surges [larger than {:.1f}%]:'
              .format(nuclear_area_fraction_surge_threshold*100))
        for (track_name, time_frame), area_rel_change in surges:
            print('  track {:3s} [@ time-point {:3d}]: nucleus area +{:.1f}%'
                  .format(track_name, time_frame, 100*area_rel_change))
        print()

    return (drops, surges)



def _find_delimiting_time_points(qtracks):
    delimiting_time_points = []
    for track_i, (_, qtrack, __) in enumerate(qtracks):
        first_last_indices = [0, -1]
        begin, end = qtrack['time_point_index'].iloc[first_last_indices]
        delimiting_time_points += [(track_i, (begin, end))]
    return sorted(delimiting_time_points, key=lambda dtp: dtp[-1][0])



def suggest_merging(qtracks,
                    min_time_points_sep,
                    max_time_points_sep,
                    displacements_reference_quantile,
                    max_displacement_in_median_displacements_sq,
                    skip_tracks_shorter_than_n_timepoints,
                    print_results,
                    print_best_matches_n):

    displ_sq_desc = _calculate_displacements_squared_descending(qtracks)
    displ_sq_desc_idx = int(displacements_reference_quantile * len(displ_sq_desc))
    refrence_in_track_d_sq = displ_sq_desc[displ_sq_desc_idx][-1]
    max_d_sq = refrence_in_track_d_sq*max_displacement_in_median_displacements_sq

    delimiting_time_points = _find_delimiting_time_points(qtracks)

    xy_col_names = ['nuc_center_x', 'nuc_center_y']
    matches = []
    for track1_index, (beg1, end1) in delimiting_time_points:
        if end1 - beg1 + 1 < skip_tracks_shorter_than_n_timepoints:
            continue
        for track2_index, (beg2, end2) in delimiting_time_points:
            if end2 - beg2 + 1 < skip_tracks_shorter_than_n_timepoints:
                continue
            overlap = (end1 >= beg2)
            too_distant_temporally = (beg2 - end1 > max_time_points_sep)
            too_close_temporally = (beg2 - end1 < min_time_points_sep)
            if overlap or too_distant_temporally or too_close_temporally:
                continue
            xy_end1 = qtracks[track1_index][1].tail(1)[xy_col_names].values[0]
            xy_beg2 = qtracks[track2_index][1].head(1)[xy_col_names].values[0]
            d_sq = (xy_beg2[0] - xy_end1[0])**2 + (xy_beg2[1] - xy_end1[1])**2
            if d_sq < max_d_sq:
                track1_name, track2_name = qtracks[track1_index][0], qtracks[track2_index][0]
                matches += [((track1_name, (beg1, end1), track2_name, (beg2, end2)), d_sq)]

    matches = sorted(matches, key=lambda z: z[-1])
    if print_results:
        if print_best_matches_n:
            print('Possible track mergers [showing{} best {}]:'
                  .format('' if len(matches) > print_best_matches_n
                          else ' at most', print_best_matches_n))
        for match_i, ((track1_name, (beg1, end1), track2_name, (beg2, end2)), d_sq) \
                in enumerate(matches):
            if match_i >= print_best_matches_n:
                break
            len1, len2 = end1 - beg1 + 1, end2 - beg2 + 1
            print('  match: [{:3d}:{:3d}] @ track {:3s} -&- track {:3s} @ [{:3d}:{:3d}]'
                  '  gap:{:2d}  displacement:{:4.1f}  lengths:{:3d}+{:3d}'
                  .format(beg1, end1, track1_name, track2_name, beg2, end2,
                          beg2 - end1 - 1, d_sq**0.5, len1, len2))
        print()

    return matches



def infer_cell_divisions(qtracks,
                         min_time_points_sep,
                         max_time_points_sep,
                         max_displacement_in_median_displacements_sq_per_step,
                         skip_tracks_shorter_than_n_timepoints,
                         print_results):
    """
    Returns
    -------
        cell_divisions : dict
            Keys of the returned dictionary are parent cell track ids (0-based),
            values are tuples containing end time point of parent cell and names of
            daughter cell tracks.
    """

    max_displacement_sq = max_time_points_sep*max_displacement_in_median_displacements_sq_per_step
    matches = suggest_merging(qtracks,
                              min_time_points_sep=min_time_points_sep,
                              max_time_points_sep=max_time_points_sep,
                              displacements_reference_quantile=0.5,
                              max_displacement_in_median_displacements_sq=\
                                      max_displacement_sq,
                              skip_tracks_shorter_than_n_timepoints=\
                                      skip_tracks_shorter_than_n_timepoints,
                              print_results=False, print_best_matches_n=0)

    matches_dict = dict()
    for (track1_name, (begin1, end1), track2_name, (begin2, end2)), d_sq in matches:
        assert end1 < begin2
        if not track1_name in matches_dict.keys():
            matches_dict[track1_name] = []
        matches_dict[track1_name] += [(track1_name, (begin1, end1),
                                       track2_name, (begin2, end2), d_sq)]

    cell_divisions = dict()
    for parent_track_name, parent_daughter_data in matches_dict.items():
        if len(parent_daughter_data) < 2:
            continue
        daugd = sorted(parent_daughter_data, key=lambda z: z[3][0]) \
                if len(parent_daughter_data) > 2 else parent_daughter_data
        parent_end = parent_daughter_data[0][1][-1]
        daughter1_track_name, daughter2_track_name = daugd[0][2], daugd[1][2]
        cell_divisions[parent_track_name] = (parent_end, daughter1_track_name, daughter2_track_name)

    if print_results:
        print('Inferred cell division events:')
        def daughter_begin(parent_track_name_, daughter_idx):
            return matches_dict[parent_track_name_][daughter_idx][3][0]
        cell_divisions_list = sorted(cell_divisions.items(), key=lambda par_dau: par_dau[0])
        for parent_track_name, (parent_end, daughter1_track_name, daughter2_track_name) \
                in cell_divisions_list:
           #print('DEVEL', parent_track_name, parent_end, daughter1_track_name,daughter2_track_name)
            print('  track {:3s} [@ time point {:3d}] --> '
                  'track {:3s} [@{:3d}] -&- track {:3s} [@{:3d}]'
                  .format(parent_track_name, parent_end,
                          daughter1_track_name, daughter_begin(parent_track_name, 0),
                          daughter2_track_name, daughter_begin(parent_track_name, 1)))

    return cell_divisions



def inspect(qtracks,
            skip_displacements=False,
            print_most_displaced_n=30,
            skip_nuclear_area=False,
            nuclear_area_fraction_drop_threshold=0.25,
            nuclear_area_fraction_surge_threshold=0.25,
            skip_merging_suggestions=False,
            merging_min_time_points_sep=0,
            merging_max_time_points_sep=5,
            merging_displacements_reference_quantile=0.25,  # descending order => 3rd quartile
            merging_max_displacement_in_ref_displacements_sq=5,
            merging_skip_tracks_shorter_than_n_timepoints=2,
            merging_print_best_matches_n=50,
            skip_cell_divisions=False,
            divisions_min_time_points_sep=1,
            divisions_max_time_points_sep=10,
            divisions_max_displacement_in_median_displacements_sq_per_step=7**2,
            divisions_skip_tracks_shorter_than_n_timepoints=2,
            print_results=True):

    # Note: displacements could be potentially shared between check_displacements_sanity()
    #       and suggest_merging() and infer_cell_divisions()

    if not skip_displacements:
        displacements = \
                check_displacements_sanity(qtracks,
                                           print_results,
                                           print_most_displaced_n)
    else:
        displacements = None


    if not skip_nuclear_area:
        nuclear_area_drops, nuclear_area_surges = \
                check_nuclear_area_consistency(qtracks,
                                               nuclear_area_fraction_drop_threshold,
                                               nuclear_area_fraction_surge_threshold,
                                               print_results)
    else:
        nuclear_area_drops, nuclear_area_surges = None, None


    if not skip_merging_suggestions:
        merge_matches = \
                suggest_merging(qtracks,
                                merging_min_time_points_sep,
                                merging_max_time_points_sep,
                                merging_displacements_reference_quantile,
                                merging_max_displacement_in_ref_displacements_sq,
                                merging_skip_tracks_shorter_than_n_timepoints,
                                print_results,
                                merging_print_best_matches_n)
    else:
        merge_matches = None


    if not skip_cell_divisions:
        cell_divisions = \
                infer_cell_divisions(qtracks,
                                     divisions_min_time_points_sep,
                                     divisions_max_time_points_sep,
                                     divisions_max_displacement_in_median_displacements_sq_per_step,
                                     divisions_skip_tracks_shorter_than_n_timepoints,
                                     print_results)
    else:
        cell_divisions = None

    if print_results:
        return None

    return (displacements, nuclear_area_drops, nuclear_area_surges, merge_matches, cell_divisions)



def print_inspection_per_track(qtracks, inspection_data):

    displacements, nuclear_area_drops, nuclear_area_surges, \
            merge_matches, cell_divisions = inspection_data

    for track_name, _, is_revised in qtracks:
        remarks = []

        track_displacements = []
        for displ_track_name, time_frame_i, d_in_median_d in displacements:
            if displ_track_name == track_name:
                track_displacements += [(time_frame_i, d_in_median_d)]
        if track_displacements:
            for time_frame_i, d_in_median_d in sorted(track_displacements,
                                                      key=lambda z: -z[-1]):
                remarks += ['  @ time point {:d}: nucleus displacement ({:.1f}x median)'\
                            .format(time_frame_i, d_in_median_d)]

        track_nuclear_area_drops = []
        for (drop_track_name, time_point_index), drop in nuclear_area_drops:
            if drop_track_name == track_name:
                track_nuclear_area_drops += [(time_point_index, drop)]
        for time_point_index, drop in track_nuclear_area_drops:
            remarks += ['  @ time point {:d}: nucleus area change ({:.1f}%)'\
                        .format(time_point_index, 100*drop)]

        track_nuclear_area_surges = []
        for (surge_track_name, time_point_index), surge in nuclear_area_surges:
            if surge_track_name == track_name:
                track_nuclear_area_surges += [(time_point_index, surge)]
        for time_point_index, surge in track_nuclear_area_surges:
            remarks += ['  @ time point {:d}: nucleus area change (+{:.1f}%)'\
                        .format(time_point_index, 100*surge)]

        remarks = sorted(remarks, key=lambda ln: ln.split()[3])

        for parent_track_name, (time_point, daugh1_track_name, daugh2_track_name) \
                in cell_divisions.items():
            if parent_track_name == track_name:
                remarks += ['  Parent of tracks {:s} and {:s} (division @ time point {:d})'\
                            .format(daugh1_track_name, daugh2_track_name, time_point)]
            elif track_name in [daugh1_track_name, daugh2_track_name]:
                remarks += ['  Daughter of track {:s} (division @ time point {:d})'\
                            .format(parent_track_name, time_point)]

        for (track1_name, _, track2_name, __), ___ in merge_matches:
            if track1_name == track_name:
                remarks += ['  Likely, could be continued as track {:s}'.format(track2_name)]
            elif track2_name == track_name:
                remarks += ['  Likely, a continuation of track {:s}'.format(track1_name)]

        if not remarks:
            print()
            print('Track {:s}'.format(track_name), end='')
            print(' -- OK' + (' and REVISED' if is_revised else ' (but NOT revised)'))
        else:
            print()
            print('Track {:s}:'.format(track_name))
            print('\n'.join(remarks))
            print('  REVISED' if is_revised else '  NOT revised.')
