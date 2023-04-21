from pathlib import Path

# If DATA_SOURCE == 'EXTERNAL', data is imported from csvs in the form as published on Zenodo 
# If DATA_SOURCE == 'INTERNAL, full data package as originally used in our lab will be loaded
DATA_SOURCE = 'EXTERNAL'

root_working_directory = Path('C:/Users/pawna/Documents/Projekty/Bern/code')  # path to the root folder of the downloaded package. You can specify any other directory on your computer. Can be pathlib.path or str
full_data_directory = root_working_directory / 'data/full_data'
images_root_directory =  root_working_directory / 'data/mapk-info-rate-extra-images'
external_data_directory = root_working_directory / 'data/Nalecz-Jawecki_et_al--Source_Data'

# Where to cache pickles with intermediate results.
# Path can be ablosute or relative to the folder containing data for a particular experiment.
# If relative, there will be a separate cache directory for each experiment.
cache_directory = root_working_directory / '_analysis_cache'

# DATA_SOURCE = 'INTERNAL'
# # root_working_directory = Path('C:/Users/pawna/Documents/Projekty/Bern/EXPORT')  # path to the root folder of the downloaded package. You can specify any other directory on your computer. Can be pathlib.path or str
# full_data_directory = Path('C:/Users/pawna/Documents/Projekty/Bern/EXPORT') / 'data/full_data'

