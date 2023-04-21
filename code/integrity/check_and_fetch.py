from pathlib import Path
import sys
from urllib import request
from zipfile import ZipFile
from progressbar import ProgressBar
import click

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent.parent))

from core.local_config import full_data_directory, external_data_directory, images_root_directory, DATA_SOURCE
from core.experiment_manager import experiments, chosen_experiments_pseudorandom, chosen_experiments_interval, chosen_experiments_interval_with_gap, map_to_official_naming


assert DATA_SOURCE in ('EXTERNAL', 'INTERNAL'), f"DATA_SOURCE must be either 'INTERNAL' or 'EXTERNAL'. Specify it in 'core/local_config.py'."

full_data_directory = Path(full_data_directory).absolute()
external_data_directory = Path(external_data_directory).absolute()
images_root_directory = Path(images_root_directory).absolute()



full_data_package_url = 'https://pmbm.ippt.pan.pl/data/mapk-info-rate-data.zip'
full_data_package_name = 'full_data'
full_data_package_size = '8.5GB'
exportable_package_url = 'https://zenodo.org/record/7472959/files/Nalecz-Jawecki_et_al-Source_Data.zip?download=1'
exportable_package_name = 'Nalecz-Jawecki_et_al--Source_Data'
exportable_package_size = '664MB'
extra_images_url = 'https://pmbm.ippt.pan.pl/data/mapk-info-rate-extra-images.zip'
extra_images_name = 'mapk-info-rate-extra-images'
extra_images_size = '108MB'


def flatten(xx):
    return [x for y in xx for x in y]


def fetch_data(url: str, path_to_store: Path, package_name: str):

    print(f"Fetching data from {url}...")
    pbar = ProgressBar()
    pbar.start()
    def reporthook(block_no, read_size, total_size): 
        try:
            pbar.update(int(block_no*read_size/total_size*100))#print(f"{block_no=}, {read_size=}, {total_size=}")
        finally:
            pass
    archive_path, _ = request.urlretrieve(url, reporthook=reporthook)
    pbar.finish()

    temporary_directory = path_to_store.parent / (path_to_store.name + '___tmp')
    print(f"Extracting files to '{temporary_directory}' from '{archive_path}' ...")
    with ZipFile(archive_path, "r") as z:
        z.extractall(temporary_directory)
    
    print(f"Copying to '{path_to_store}' ...")
    assert (temporary_directory / package_name).exists(), f"Folder {package_name} not found in the downloaded package. Check the package name and the url"
    (temporary_directory / package_name).rename(path_to_store)
    print(f"Removing '{temporary_directory}'...")
    temporary_directory.rmdir()
    print('done.')




def check_full_package():
    if not full_data_directory.exists():
        if click.confirm(f'The scripts are running in INTERNAL mode and therefore require the full data package. No source data was found in "{full_data_directory}". Do you want to fetch it from "{full_data_package_url}" ({full_data_package_size})?\n(If you have already downloaded the full data package, type "N" to abort and specify the path to your package in "core/local_config.py")\n', default=True):
            fetch_data(full_data_package_url, full_data_directory, full_data_package_name)

    assert full_data_directory.exists(), 'Full package directory does not exist'
    for experiment_name in chosen_experiments_pseudorandom + chosen_experiments_interval + chosen_experiments_interval_with_gap:
        experiment = experiments[experiment_name]
        assert (experiment['working_directory'] / (experiment['directory'])).exists(), f"Directory {(experiment['directory'])} not found in {experiment['working_directory']}"
    print('Full data package directory OK')


def check_exportable_package():

    if not external_data_directory.exists():
        if click.confirm(f'No data was found in "{external_data_directory}". Do you want to fetch it from "{exportable_package_url}" ({exportable_package_size})?\n(If you have already downloaded the package, type "N" to abort and specify the path to the package in "core/local_config.py")\n', default=True):
            fetch_data(exportable_package_url, external_data_directory, 'Nalecz-Jawecki_et_al--Source_Data')

    assert external_data_directory.exists(), 'CSV root directory does not exist'
    for experiment_type in ('binary_encoding', 'interval_encoding', 'interval_encoding_with_minimal_gap'):
        assert (external_data_directory / experiment_type).exists(), f"CSV root directory ({external_data_directory}) does not contain subdirectory {experiment_type}"
    for experiment_name in chosen_experiments_pseudorandom + chosen_experiments_interval + chosen_experiments_interval_with_gap:
        experiment = experiments[experiment_name]
        assert (experiment['working_directory'] / (experiment['directory'] + '.csv')).exists(), f"File {(experiment['directory'] + '.csv')} not found in {experiment['working_directory']}"
    print('Exportable package directory OK')


def check_extra_images():
    if not images_root_directory.exists():
        if click.confirm(f'To create this figure, a package with cell images is required. Appropriate data was not found in "{images_root_directory}". Do you want to fetch it from "{extra_images_url}" ({extra_images_size})?\n(If you have already downloaded the image package, type "N" to abort and specify the path to the package in "core/local_config.py")\n', default=True):
            fetch_data(extra_images_url, images_root_directory, extra_images_name)

    
    assert images_root_directory.exists(), 'Image root directory does not exist'
    for experiment_type in ('interval_encoding',):
        assert (images_root_directory / experiment_type).exists(), f"Images root directory ({images_root_directory}) does not contain subdirectory {experiment_type}"
    for experiment_type, experiment_name in [('interval_encoding', 'min3_mean30')]:
        official_name = map_to_official_naming(experiment_name)
        assert (images_root_directory / experiment_type / (official_name + 'None.csv')).exists(), f"File {(official_name + 'None.csv')} not found in {images_root_directory / experiment_type}"
        assert (images_root_directory / experiment_type / official_name).is_dir(), f"Directory {images_root_directory / experiment_type / official_name} must exist"
        for file in [
            *flatten([(f'b1_t0{t}_C1.tif', f'b1_t0{t}-img_quant.csv', f'b1_t0{t}-nuclei.txt') for t in range(621, 637)]),
            'tracks.csv']:
            assert (images_root_directory / experiment_type / official_name / file).exists(), f"File {file} not found in {images_root_directory / experiment_type / official_name}"
    print('Images directory OK')
        

def check_and_fetch_necessary(requires_fig1cd_package=False):
    print('Checking data availability...', flush=True)
    if DATA_SOURCE == 'EXTERNAL':
        check_exportable_package()
        if requires_fig1cd_package:
            check_extra_images()
    else:
        check_full_package()
    


