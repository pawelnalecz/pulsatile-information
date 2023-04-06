import sys
import numpy as np
import pandas as pd
from pathlib import Path
import click

sys.path.append(str(Path(__file__).parent.parent))

from core import experiment_manager, factory
from core.steps import extract_track_information, preselect_tracks
from core.local_config import DATA_SOURCE
from figures.local_config import figure_output_path
from integrity import check_and_fetch

assert DATA_SOURCE == 'INTERNAL' or click.confirm(f"Current data source is set to {DATA_SOURCE}. Exporting data from already exported data seems useless. Change DATA_SOURCE to 'INTERNAL' to use original pickles instead, or type 'Y' to proceed anyway.", default=False)

check_and_fetch.check_and_fetch_necessary()

output_path = Path(figure_output_path).absolute() / 'data_export_package'
output_path.mkdir(parents=True, exist_ok=True)


Q_fields = ['isBlink', 'nuc_area', 'nuc_center_x', 'nuc_center_y', 'nuc_H2B_intensity_mean', 'nuc_ERKKTR_intensity_mean', 'img_H2B_intensity_mean', 'img_ERKKTR_intensity_mean', 'Q2', 'Q3backw']

for group_it,(title, regular, take_tracks, experiments) in enumerate((

    ('binary_encoding', True, 500,
        experiment_manager.chosen_experiments_pseudorandom,
    ),
    ('interval_encoding', False, 500,
        experiment_manager.chosen_experiments_interval,
    ),
    ('interval_encoding_with_minimal_gap', False, 500,
        experiment_manager.chosen_experiments_interval_with_gap,
    ),
    # ('mapk-info-rate-extra-images', False, None,
    #     ['min3_mean30'],
    # ),
)): 
    for it_experiment, experiment in enumerate(experiments):
        parameters = {
                    **experiment_manager.default_parameters,
                    **experiment_manager.experiments[experiment],
                    'theoretical_parameters': experiment_manager.theoretical_parameters[experiment],
                    'take_tracks': 'preselected',#range(50),#range(1000),#
                    'trim_end': (-1, int(np.floor(experiment_manager.theoretical_parameters[experiment]['min'] + experiment_manager.theoretical_parameters[experiment]['exp_mean']))),
        

                    **({
                        'yesno': True,
                    } if regular else {}),
                    
                    'vivid_track_criteria': [
                        ('', 'index', 'lt', 500),
                        ('std_dQ2', 'rank', 'gt', 0.2),
                    ],
                }
        chain = factory.quantify_tracks(parameters).step(extract_track_information).step(preselect_tracks)
        
        group_output_path = output_path / title
        group_output_path.mkdir(parents=True, exist_ok=True)

        
        Q = chain.load_file('raw_tracks')[0:take_tracks]#pulses.load_shuttletracker_data(parameters['directory'], n_tracks=parameters['n_tracks'])[0:take_tracks]
        quantified_tracks = chain.load_file('quantified_tracks')[0:take_tracks] 
        vivid_tracks = chain.load_file('vivid_tracks')

        actual_take_tracks = len(Q) if take_tracks == None else take_tracks

        print("Concatenating data", end=' ...', flush=True)
        Q_df = pd.concat(Q, keys=range(actual_take_tracks), names=['track_id', 'time_in_minutes'])
        quantified_tracks_df = pd.concat(quantified_tracks, keys=range(actual_take_tracks), names=['track_id', 'time_in_minutes'])
        print("Exporting to csv", end=' ...', flush=True)
        print(Q_df.join(quantified_tracks_df).columns)
        Q_df.join(quantified_tracks_df)[Q_fields].assign(
                Q2=lambda x: -x['Q2'],
                ERKKTR_translocation=lambda x: 1-x['Q3backw'],
                is_preselected=lambda x: x.index.get_level_values('track_id').isin(vivid_tracks)
            ).rename({
                    'Q2': 'nuc_ERKKTR_intensity_mean_normalized_with_image', 
                    'Q3backw': 'nuc_ERKKTR_intensity_mean_normalized_with_image_and_history', 
                    'isBlink': 'is_light_pulse'
                }, axis='columns'
            ).reset_index().to_csv(group_output_path / (experiment_manager.map_to_official_naming(experiment) + (str(take_tracks) if take_tracks != 500 else '') +'.csv'), float_format="{:.7g}".format)
        print("done.")
        print(f"CSV file saved to {group_output_path / (experiment_manager.map_to_official_naming(experiment) + (str(take_tracks) if take_tracks != 500 else '') + '.csv')}")
