#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:24:06 2019

@author: adam

Creates figure showing displacements of one mitochondria in one trial.
"""
import yaml
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import distance
import warnings

experiment_id = 'SSN_116_001'
analyzed_data_dir = ('/Users/adam/Documents/SenseOfTouchResearch/'
                     'SSN_ImageAnalysis/AnalyzedData/' + experiment_id)
"""
#strain_file = analyzed_data_dir + '/strain.yaml'
#with open(strain_file, 'r') as yamlfile:
#    strain_results = yaml.load(yamlfile)
"""

mito_locations_file = analyzed_data_dir + '/trackpyBatchResults.yaml'
with open(mito_locations_file, 'r') as yamlfile:
    mito_locations = yaml.load(yamlfile)

metadata_file = analyzed_data_dir + '/metadata.yaml'
with open(metadata_file, 'r') as yamlfile:
    metadata = yaml.load(yamlfile)

linked_mitos = pd.DataFrame.from_dict(mito_locations, orient='index')
mito_locations = linked_mitos.loc[:, ['frame', 'particle', 'x', 'y', 'z']]


def get_pressure(row):
    frame_num = int(row['frame'])
    pressure = metadata['pressure_kPa'][frame_num]
    return pressure


mito_locations['pressure'] = mito_locations.apply(
        lambda row: get_pressure(row), axis=1)


def get_rest_location(particle, axis):

    x_rest = mito_locations.loc[(mito_locations['particle'] == particle) &
                                (mito_locations['pressure'] == 0)]['x'].mean()
    y_rest = mito_locations.loc[(mito_locations['particle'] == particle) &
                                (mito_locations['pressure'] == 0)]['y'].mean()
    z_rest = mito_locations.loc[(mito_locations['particle'] == particle) &
                                (mito_locations['pressure'] == 0)]['z'].mean()
    if axis == 'x':
        rest_location = x_rest
    elif axis == 'y':
        rest_location = y_rest
    elif axis == 'z':
        rest_location = z_rest
    else:
        raise ValueError("Invalid axes value")

    return rest_location


mito_locations['x_rest'] = mito_locations.apply(
        lambda row: get_rest_location(row['particle'], 'x'), axis=1)
mito_locations['y_rest'] = mito_locations.apply(
        lambda row: get_rest_location(row['particle'], 'y'), axis=1)
mito_locations['z_rest'] = mito_locations.apply(
        lambda row: get_rest_location(row['particle'], 'z'), axis=1)

mito_pairs = pd.DataFrame(columns=['x_dist', 'y_dist', 'z_dist', 'x_1', 'x_2',
                                   'y_1', 'y_2', 'z_1', 'z_2', 'total_dist'
                                   'particle_1', 'particle_2', 'frame',
                                   'pair_id'])
mito_pairs_dicts = []
for frame in mito_locations.frame.unique():
    this_frame = mito_locations.loc[(mito_locations['frame'] == frame)]
    this_frame.sort_values(by=['y'], inplace=True)
    this_frame.reset_index(inplace=True)
    # make dataframe of adjacent mitos
    for this_particle in range(mito_locations.particle.nunique() - 1):
        x_1 = this_frame.iloc[this_particle]['x']
        x_2 = this_frame.iloc[this_particle + 1]['x']
        x_dist = abs(x_2 - x_1)

        y_1 = this_frame.iloc[this_particle]['y']
        y_2 = this_frame.iloc[this_particle + 1]['y']
        y_dist = abs(y_2 - y_1)

        z_1 = this_frame.iloc[this_particle]['z']
        z_2 = this_frame.iloc[this_particle + 1]['z']
        z_dist = abs(z_2 - z_1)

        total_dist = distance.euclidean([x_1, y_1, z_1], [x_2, y_2, z_2])

        particle_1 = int(this_frame.iloc[this_particle]['particle'])
        particle_2 = int(this_frame.iloc[this_particle + 1]['particle'])

        mito_pairs_dicts.append({'x_dist': x_dist, 'x_1': x_1, 'x_2': x_2,
                                 'y_dist': y_dist, 'y_1': y_1, 'y_2': y_2,
                                 'z_dist': z_dist, 'z_1': z_1, 'z_2': z_2,
                                 'particle_1': particle_1,
                                 'particle_2': particle_2,
                                 'total_dist': total_dist, 'frame': frame,
                                 'pair_id': this_particle})
mito_pairs = pd.DataFrame(mito_pairs_dicts)
mito_pairs['pressure'] = mito_pairs.apply(lambda row: get_pressure(row),
                                          axis=1)


def get_rest_distance(pair, axis):
    euclidean_rest_dist = mito_pairs.loc[(mito_pairs['pair_id'] == pair) &
                                         (mito_pairs['pressure'] == 0)][
                                                      'total_dist'].mean()
    x_rest_dist = mito_pairs.loc[(mito_pairs['pair_id'] == pair) &
                                 (mito_pairs['pressure'] == 0)][
                                         'x_dist'].mean()
    y_rest_dist = mito_pairs.loc[(mito_pairs['pair_id'] == pair) &
                                 (mito_pairs['pressure'] == 0)][
                                         'y_dist'].mean()
    z_rest_dist = mito_pairs.loc[(mito_pairs['pair_id'] == pair) &
                                 (mito_pairs['pressure'] == 0)][
                                         'z_dist'].mean()
    if axis == 'euclid':
        rest_distance = euclidean_rest_dist
    elif axis == 'x':
        rest_distance = x_rest_dist
    elif axis == 'y':
        rest_distance = y_rest_dist
    elif axis == 'z':
        rest_distance = z_rest_dist
    else:
        raise ValueError("Invalid axes value")

    return rest_distance


mito_pairs['rest_dist'] = mito_pairs.apply(
        lambda pair: get_rest_distance(pair['pair_id'], 'euclid'), axis=1)
mito_pairs['x_rest_dist'] = mito_pairs.apply(
        lambda pair: get_rest_distance(pair['pair_id'], 'x'), axis=1)
mito_pairs['y_rest_dist'] = mito_pairs.apply(
        lambda pair: get_rest_distance(pair['pair_id'], 'y'), axis=1)
mito_pairs['z_rest_dist'] = mito_pairs.apply(
        lambda pair: get_rest_distance(pair['pair_id'], 'z'), axis=1)

mito_pairs['strain'] = ((mito_pairs['total_dist'] - mito_pairs['rest_dist'])
                        / mito_pairs['rest_dist'])
mito_pairs['x_strain'] = ((mito_pairs['x_dist'] - mito_pairs['x_rest_dist'])
                          / mito_pairs['x_rest_dist'])
mito_pairs['y_strain'] = ((mito_pairs['y_dist'] - mito_pairs['y_rest_dist'])
                          / mito_pairs['y_rest_dist'])
mito_pairs['z_strain'] = ((mito_pairs['z_dist'] - mito_pairs['z_rest_dist'])
                          / mito_pairs['z_rest_dist'])


def make_plot_vs_ycoord(dataframe, x, y):
    if 'strain' in y:
        kwargs = {'drawstyle': 'steps'}
    else:
        kwargs = {'kind': 'scatter'}
    ax = dataframe.loc[(dataframe['pressure'] == 0)].plot(
        y=y, x=x,  color='red', **kwargs)
    try:
        dataframe.loc[(dataframe['pressure'] == 300)].plot(
            y=y, x=x, color='green', ax=ax, **kwargs)
    except ValueError:
        if 300 not in dataframe['pressure'].unique():
            warnings.warn('No actuation in this trial')

    bbox = ax.get_window_extent().transformed(
        ax.get_figure().dpi_scale_trans.inverted())
    fig = ax.get_figure()
    xmin, xmax = ax.get_xlim()
    fig.set_size_inches((xmax - xmin) / ax.get_figure().dpi, bbox.height)

    return ax


# Plot of strain vs. y coordinate
strain_ax = make_plot_vs_ycoord(mito_pairs, 'y_1', 'strain')
strain_ax.get_figure().savefig('/Users/adam/Downloads/strain_fig.png')

# Plot of x_strain vs. y coordinate
x_strain_ax = make_plot_vs_ycoord(mito_pairs, 'y_1', 'x_strain')
x_strain_ax.get_figure().savefig('/Users/adam/Downloads/x_strain_fig.png')

# Plot of xz displacement vs. y coordinate
mito_locations['xz_disp'] = mito_locations.apply(
        lambda row: distance.euclidean(
                [row.x, row.z], [row.x_rest, row.z_rest]), axis=1)
xz_disp_ax = make_plot_vs_ycoord(mito_locations, 'y', 'xz_disp')
xz_disp_ax.get_figure().savefig('/Users/adam/Downloads/xz_disp_fig.png')

# Plot of total displacement vs. y coordinate
mito_locations['xyz_disp'] = mito_locations.apply(
        lambda row: distance.euclidean(
                [row.x, row.y, row.z],
                [row.x_rest, row.y_rest, row.z_rest]), axis=1)
xyz_disp_ax = make_plot_vs_ycoord(mito_locations, 'y', 'xyz_disp')
xyz_disp_ax.get_figure().savefig('/Users/adam/Downloads/xyz_disp_fig.png')
