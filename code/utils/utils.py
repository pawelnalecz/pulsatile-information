# utils.py

import os
from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as clrs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import re

from core.step_manager import Chain


# def complementary_pseudorandom_experiment(experiment):
#     pos_text = re.search('pos[0-9]+_', experiment).group(0)
#     pos = int(pos_text[3:-1])
#     complementary_pos_text = f'pos{11-pos:02d}_'
#     return experiment.replace(pos_text, complementary_pos_text)

def another_covariant_pseudorandom_experiment(experiment, available_experiments):
    pos_text = re.search('pos[0-9]+_', experiment).group(0)
    pos = int(pos_text[3:-1])

    has10trials = experiment.replace(pos_text,  f'pos{1:02d}_') in available_experiments
    another_pos = (
        1 if pos == 5 and has10trials else
        6 if pos == 10 and has10trials else
        2 if pos == 5 and not has10trials else
        6 if pos == 9 and not has10trials else
        pos+1
    )
                     
    complementary_pos_text = f'pos{another_pos:02d}_'
    return experiment.replace(pos_text, complementary_pos_text)


def list_without(l : Iterable, *exclude_elements) -> list:
    return list(filter(lambda x: not(x in exclude_elements), l))


def show_slice_gen(chain, with_results=False):
    quantified_tracks =  chain.load_file('quantified_tracks')
    if with_results:
        results = chain.load_file('prediction_results')

    def show_slice(track_ids, slice_no, fields=None, margin=0, agg=None, ax=plt.gca(), point_plot_kwargs=dict(), gauge=False, with_results=False, **kwargs):
        fields = fields or chain.parameters['fields_for_learning']

        def plot_trajectory(df):
            if gauge: df = df - df.loc[slice_no]
            index = df.index[(df.index >=slice_no - margin) & ( df.index < slice_no + chain.parameters['slice_length'] + margin)]
            the_slice = df.reindex(index)\
                .reset_index(drop=True)\
                .assign(time=lambda _: index - slice_no).set_index('time')
            the_slice.plot(ax=ax, **kwargs)
            the_slice.reindex([0, (chain.parameters['slice_length']-1)*(index[1]-index[0])]).plot(ax=ax, ls='', marker='o', **point_plot_kwargs)

            
        if not agg:
            for track_id in track_ids:
                if (slice_no not in quantified_tracks[track_id].index) or (slice_no + chain.parameters['slice_length'] not in quantified_tracks[track_id].index): continue
                plot_trajectory(quantified_tracks[track_id][fields])
                if with_results:
                    index = quantified_tracks[track_id].index[(quantified_tracks[track_id].index >=slice_no) & ( quantified_tracks[track_id].index < slice_no + chain.parameters['slice_length'])]
                    for t in index:
                        plt.annotate(', '.join(str(x) for x in results[(results.index.get_level_values('track_id') == track_id) & (results.index.get_level_values('slice_no') == t)]['y_pred']), (t-slice_no, quantified_tracks[track_id].loc[t][fields].iloc[0]), fontsize='x-small')
        else: 
            plot_trajectory(pd.concat({track_id: quantified_tracks[track_id][fields] for track_id in track_ids}).groupby(level=1).agg(agg))

        # plt.annotate(str(track_id), (0, the_slice.loc[0]))

    return show_slice

def show_cell_in_image_gen(chain: Chain, colormaps=None, marker='+', shift_marker=(0,0), scatter_dict={}, img_path=None):
    
    curdir  = os.curdir
    os.chdir(chain.working_directory)

    img_path = img_path or chain.parameters['working_directory']

    assert 'raw_tracks' in chain.files, 'Showing cells requires raw tracks to obtain nuclei positions'
    Q = chain.load_file('raw_tracks')

    colormaps= colormaps or [
        plt.get_cmap('Reds'),
        plt.get_cmap('YlOrBr'),
        clrs.ListedColormap([(0., 0., 1., x/n) for n in [50] for x in range(n+1)])
    ]
    os.chdir(curdir)
    scatter_dict_global = scatter_dict


    def show_cell_in_image(track_id, T, channels=[1, 2], window_half_size=40, shift_in_time=0, ax=None, norm_quantiles=None, norm_absolute=None, scatter_dict={}):

        if not (norm_absolute or norm_quantiles):
            norm_quantiles = (.1, .8)

        if ax: 
            ax_old = plt.gca()
            plt.sca(ax)
        x,y = map(lambda x: int(np.round(x)), Q[track_id][['nuc_center_x', 'nuc_center_y']].loc[T])
        

        alpha=1
        files = os.listdir(img_path / chain.parameters['directory'])
        for channel in channels:
            filename=''
            for file in files:
                if '.tif' in file and f'c{channel}' in file.lower() and (f'{T+shift_in_time:04d}' in file or f'_T{T+shift_in_time:03d}' in file):
                    filename = img_path / chain.parameters['directory'] / file
                    break

            assert filename, f'File pattern not matched for T = {T:d} among ' + str([file for file in files if '.tif' in file])
            img = plt.imread(filename)
            normalization = clrs.Normalize(np.quantile(img, norm_quantiles[0]), np.quantile(img, norm_quantiles[1])) if norm_quantiles else clrs.Normalize(norm_absolute[0], norm_absolute[1])
            plt.imshow(img[ max(y-window_half_size, 0):min(y+window_half_size+1, img.shape[1]), max(x-window_half_size, 0):min(x+window_half_size+1, img.shape[0])], 
            extent=(max(x-window_half_size, 0), min(x+window_half_size+1, img.shape[0]), min(y+window_half_size+1, img.shape[1]), max(y-window_half_size, 0)), 
            cmap=colormaps[channel], alpha = alpha, norm=normalization)
            alpha *= .4
        plt.scatter(x+shift_marker[0], y+shift_marker[1], marker=marker, **{**scatter_dict_global, **scatter_dict})
        if ax:
            plt.sca(ax_old)
        return (x,y)
            # plt.imshow(img_nuc[ y-window_half_size:y+window_half_size+1, x-window_half_size:x+window_half_size+1], extent=(y-window_half_size, y+window_half_size+1, x-window_half_size, x+window_half_size+1), cmap=clrs.ListedColormap([(0., 0., 1., x/n) for n in [4] for x in range(n+1)]), alpha=.4)
    return show_cell_in_image

def show_cells_with_labels_gen(chain, colormaps=None, image_path=None):
    curdir  = os.curdir
    os.chdir(chain.working_directory)

    image_path = (image_path or chain.parameters['working_directory']) / chain.parameters['directory']
    
    assert 'raw_tracks' in chain.files, 'Showing cells requires raw tracks to obtain nuclei positions'
    Q = chain.load_file('raw_tracks')

    colormaps= colormaps or [
        plt.get_cmap('Reds'),
        plt.get_cmap('YlOrBr'),
        clrs.ListedColormap([(0., 0., 1., x/n) for n in [50] for x in range(n+1)])
    ]
    os.chdir(curdir)

    def show_cells_cells_with_labels(T, channels=[1, 2], shift_in_time=0, ax=None, norm_quantiles=(.1, .8)):
        if ax: 
            ax_old = plt.gca()
            plt.sca(ax)


        alpha=1
        files = os.listdir(image_path)
        for channel in channels:
            filename=''
            for file in files:
                if '.tif' in file and f'c{channel}' in file.lower() and (f'{T+shift_in_time:04d}' in file or f'_T{T+shift_in_time:03d}' in file):
                    filename = image_path / file
                    break
            assert filename, f'File pattern not matched for T = {T:d} among ' + str([file for file in files if '.tif' in file])
            img = plt.imread(filename)
            plt.imshow(img, 
            cmap=colormaps[channel], alpha = alpha, norm=clrs.Normalize(np.quantile(img, norm_quantiles[0]), np.quantile(img,norm_quantiles[1])))
            alpha *= .4
        if ax:
            plt.sca(ax_old)

        for track_id in range(len(Q)):
            if T in Q[track_id].index:
                xx,yy = map(lambda x: int(np.round(x)), Q[track_id][['nuc_center_x', 'nuc_center_y']].loc[T])
                plt.annotate(f"{track_id:d}", (xx,yy), fontsize='xx-small', horizontalalignment='center', verticalalignment='center', color='grey')
                # plt.imshow(img_nuc[ y-window_half_size:y+window_half_size+1, x-window_half_size:x+window_half_size+1], extent=(y-window_half_size, y+window_half_size+1, x-window_half_size, x+window_half_size+1), cmap=clrs.ListedColormap([(0., 0., 1., x/n) for n in [4] for x in range(n+1)]), alpha=.4)
    return show_cells_cells_with_labels


def draw_contour_gen(chain: Chain, image_path=None):
    curdir  = os.curdir
    os.chdir(chain.working_directory)
    
    image_path = (image_path or chain.parameters['working_directory']) / chain.parameters['directory']
    
    assert 'raw_tracks' in chain.files, 'Showing cells requires raw tracks to obtain nuclei positions'
    Q = chain.load_file('raw_tracks') #if DATA_SOURCE == 'EXTERNAL' else pulses.load_shuttletracker_data(chain.parameters['directory'], n_tracks=chain.parameters['n_tracks'])

    
    assert 'nuc_nucleus_id' in Q[0].columns or (image_path / 'tracks.csv').exists(), f"No source for nuclei ids for {chain.parameters['directory']} in {(image_path / 'tracks.csv')}"
    nuc_ids = (
        pd.concat((track['nuc_nucleus_id'] for track in Q), names=['track_id'], keys=range(len(Q))) if 'nuc_nucleus_id' in Q[0].columns 
        else pd.read_csv(image_path / 'tracks.csv').set_index(['track_index','time_point_index'])['nucleus_index'] 
    )

    print(nuc_ids)

    # Q = pulses.load_shuttletracker_data(chain.parameters['directory'], n_tracks=chain.parameters['n_tracks'])
    os.chdir(curdir)

    def draw_contour(track_id, T, shift_in_time = 0, plotdict={}):
        if (track_id, T) not in nuc_ids:
            print(f"WARNING: Track {track_id} has no nucleus in frame {T}")
            return
        nuc_no = nuc_ids.loc[track_id, T]
        files = os.listdir(image_path)
        filename=''
        for file in files:
            if '-nuclei.txt' in file and (f'{T+shift_in_time:04d}' in file or f'_T{T+shift_in_time:03d}' in file):
                filename = image_path / file

        assert filename, f'File pattern not matched for T = {T:d} ({T+shift_in_time:d}) among ' + str([file for file in files if '-nuclei.txt' in file])
                
        with open(filename, 'r') as file:
            lines = file.readlines()
            the_line_no = None
            for it_line, line in enumerate(lines):
                if line == f'Nucleus {nuc_no:d}\n' or line == f"Improper-nucleus {nuc_no:d}\n": 
                    the_line_no = it_line + 3
                    break
            assert the_line_no, f"Contour for nucleus {nuc_no:d} at T={T:d} not found!"
            the_line = lines[the_line_no][8:-1]
            points = [tuple(int(x) for x in pt.split(',')) for pt in the_line.split('; ') if pt]
            plt.plot(*zip(*(points +[points[0]])), **plotdict)
    return draw_contour


def show_in_pca_domain(data, pca_base_data = None, base_alpha=1, **kwargs):
    pca_base_data = pca_base_data or data
    pca = PCA(n_components=2)
    pca.fit(pca_base_data)
    plt.scatter(list(zip(*pca.transform(data)))[0], list(zip(*pca.transform(data)))[1], s=10, alpha=base_alpha/np.sqrt(len(data)), **kwargs)
    return pca

def show_in_tsne_domain(data, target=None, base_alpha=1, **kwargs):
    tsne = TSNE(n_components=2, init='pca', verbose=True)
    transformed_data = tsne.fit_transform(data, target)
    plt.scatter(list(zip(*transformed_data))[0], list(zip(*transformed_data))[1], s=10, alpha=base_alpha/np.sqrt(len(data)), **kwargs)
    return tsne


