This package was created by Paweł Nałęcz-Jawecki and Marek Kochańczyk 
at IPPT PAN and contains scripts used for creation of figures 
used in "The MAPK/ERK channel capacity exceeds 6 bit/hour" 
by Paweł Nałęcz-Jawecki et al., 2023.

Available under CC BY-SA 4.0 Creative Commons Licence (Attribution-ShareAlike 4.0 International).



Quick usage guide:

=> To create a figure, run the proper script from 'code/figures' directory.
To do this, open your terminal, change directory to the 'code' directory 
next to which you found this readMe, and type 'python figures/fig1B.py'.
	
=> Figures will be stored in a location specified in 
'code/figures/local_config.py'.

=> The figures need data. The scripts can use either of the two data sets:
    a) EXTERNAL: Compact package published on Zenodo: 
https://doi.org/10.5281/zenodo.7472959 (default) (664MB)
    b) INTERNAL: Full data set used originally in our lab to create the figures https://pmbm.ippt.pan.pl/data/mapk-info-rate-data.zip (8.5GB)
   Data source can be specified in 'code/core/local_config.py'. 
The proper package will be downloaded automatically to your 
external_data_directory or full_data_directory, respectively, 
as specified in 'code/core/local_config.py'. 
Alternatively, you can manually download the data, unzip them 
and provide the path to the unzipped folder in the config file.

=> In addition to quantified tracks, figures 1C and 1D require 
several raw images. On attempt to run the fig1CD.py script, 
the images will be automatically downloaded and stored to directory specified by images_root_directory. 
Again, you can download the data manually from 
https://pmbm.ippt.pan.pl/data/mapk-info-rate-extra-images.zip (108MB)

=> If you want to recreate the data set published on Zenodo, run 'figures/export_data.py'.

=> The scripts cache partial results in '_analysis_cache' folder 
which will make a rerun (significantly) faster.


