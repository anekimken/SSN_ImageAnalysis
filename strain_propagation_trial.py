#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:24:53 2019

@author: Adam Nekimken
"""

import os
import pathlib
import datetime
import warnings
from typing import Tuple
import glob
import numpy as np
import pims
import pandas as pd
from nd2reader import ND2Reader
from scipy.spatial import distance
import yaml
import matplotlib.pyplot as plt
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import trackpy as tp

# warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore",
                        message=("Using a non-tuple sequence for "
                                 "multidimensional indexing is deprecated"))
warnings.filterwarnings("ignore", message="Reported average frame interval")


class StrainPropagationTrial(object):
    """Class for analyzing one trial of strain propagation experiment

    Loads previously analyzed data, by default loads images, by default
    loads metadata from file if available. Contains functions for
    using particle-finding algorithm from Crocker–Grier, manipulating metadata,
    and calculating strain.

    Args:
        None

    Attributes:
        filename: A string indicating where the image file is located
        load_images: A bool indicating if images should be loaded on init
        overwrite_metadata: A bool indicating to overwrite yaml metadata file
        STATUSES: A list of possible statuses for a trial
        experiment_id: A string identifying this particular trial
        analyzed_data_location: A pathlib.Path with  location of analyzed data
        metadata_file_path: A pathlib.Path with absolute path to metadata.yaml
        param_test_history_file: A pathlib.Path to parameter history file
        batch_data_file: A pathlink.Path to latest results from particle find
        batch_history_file: A pathlib.Path to parameter history file for batch
        unlinked_particles_file: A pathlib.Path to results before linking
        brightfield_file: A pathlib.Path to brightfield image of FOV for trial
        metadata: A dictionary containing all the metadata
        image_array: Numpy array with image data. Can be None if not loaded
        mito_candidates: Dataframe with results of analyzing one timepoint
        linked_mitos: Dataframe with results of analyzing all timepoints
        latest_test_params: A dictionary with the last set of  parameters used
        strain: A Numpy array containing calculated strain for the trial
        ycoords_for_strain: A numpy array with y coordinates matching strain

    """
    STATUSES = ['No metadata.yaml file',
                'No analysis status yet',
                'Not started',
                'Testing parameters',
                'Testing parameters for batch',
                'Strain calculated',
                'Failed - too close to edge',
                'Failed - not bright enough',
                'Failed - too much movement',
                'Failed - neuron overlap']

    def __init__(self):
        self.metadata = None
        self.mito_candidates = None
        self.linked_mitos = None
        self.latest_test_params = None
        self.default_test_params = {'gaussian_width': 3,
                                    'particle_z_diameter': 21,
                                    'particle_xy_diameter': 15,
                                    'brightness_percentile': 50,
                                    'min_particle_mass': 600,
                                    'bottom_slice': 0,
                                    'top_slice': 2,
                                    'time_point': 1,
                                    'tracking_seach_radius': 20,
                                    'last_timepoint': 11}
        with open('config.yaml', 'r') as config_file:
            self.file_paths = yaml.safe_load(config_file)

    def load_trial(self,
                   filename: str,
                   load_images: bool = True,
                   overwrite_metadata: bool = False) -> Tuple[np.array, dict]:
        """Loads the data and metadata for this trial from disk.

        Loads the data needed for analyzing the trial. Also handles
        reading/writing of the metadata yaml file. First checks if the only
        thing that needs to be loaded is the metadata, since this is fastest.
        Otherwise, loads the image file, then either writes a new metadata
        file with data from the image file and Google Drive, or loads the
        existing metadata from a yaml file. It takes a long time to load a
        large amount of data, so the goal here is to load as little data
        as possible.

        Args:
            filename: A string indicating where the image file is located
            load_images: A bool indicating if images should be loaded on init
            overwrite_metadata: A bool indicating to overwrite  metadata file

        Returns:
            image_array (np.array): Image data. Can be empty if not loaded
            metadata (dict): All the metadata

        """

        self.filename = filename
        self.load_images = load_images
        self.overwrite_metadata = overwrite_metadata

        # Get the experiment ID from the filename and load the data
        basename = os.path.basename(self.filename)
        self.experiment_id = os.path.splitext(basename)[0]
        self.analyzed_data_location = pathlib.Path(
                self.file_paths['analysis_dir'] + 'AnalyzedData/' +
                self.experiment_id + '/')
#        self.analyzed_data_location = pathlib.Path(
#                '/Users/adam/Documents/SenseOfTouchResearch/'
#                'SSN_ImageAnalysis/AnalyzedData/' + self.experiment_id + '/')
        self.metadata_file_path = self.analyzed_data_location.joinpath(
                 'metadata.yaml')
        self.param_test_history_file = self.analyzed_data_location.joinpath(
                 'trackpyParamTestHistory.yaml')
        self.batch_data_file = self.analyzed_data_location.joinpath(
                 'trackpyBatchResults.yaml')
        self.batch_history_file = self.analyzed_data_location.joinpath(
                'trackpyBatchParamsHistory.yaml')
        self.unlinked_particles_file = self.analyzed_data_location.joinpath(
                'unlinkedTrackpyBatchResults.yaml')
        data_dir = os.path.dirname(self.filename)
        bf_filename = glob.glob(data_dir + '/' +
                                self.experiment_id + '*_bf.nd2')
        if len(bf_filename) > 1:
            warnings.warn('Found more than one brightfield image')
        self.brightfield_file = None
        try:
            self.brightfield_file = pathlib.Path(bf_filename[0])
        except IndexError:
            self.brightfield_file = None

        # Create directory if necessary
        if not self.analyzed_data_location.is_dir():
            self.analyzed_data_location.mkdir()

        # Load analyzed data
        if self.batch_data_file.is_file():
            with open(self.batch_data_file, 'r') as yamlfile:
                linked_mitos_dict = yaml.load(yamlfile)
                self.linked_mitos = pd.DataFrame.from_dict(
                        linked_mitos_dict, orient='index')
        if self.unlinked_particles_file.is_file():
            with open(self.unlinked_particles_file, 'r') as yamlfile:
                unlinked_mitos_dict = yaml.load(yamlfile)
                self.mitos_from_batch = pd.DataFrame.from_dict(
                        unlinked_mitos_dict, orient='index')

        if (self.load_images is False and  # don't load images
                self.overwrite_metadata is False and  # don't overwrite
                self.metadata_file_path.is_file() is True):  # yaml exists
            # only load metadata in this case
            self.metadata = self.load_metadata_from_yaml()
            self.image_array = np.empty((11, 50, 1200, 600))
            self.image_array[:] = np.nan

        else:
            self.image_array, images = self._load_images_from_disk()
            # If we don't have a yaml metadata file yet, we need to get
            # some information from Google Drive and build a yaml file
            if (self.metadata_file_path.is_file() is False or  # no yaml yet
                    self.overwrite_metadata is True):
                self.metadata = self._retrieve_metadata(images)
                if 'analysis_status' not in self.metadata:
                    self.metadata['analysis_status'] = 'Not started'
                self.write_metadata_to_yaml(self.metadata)
            else:
                # if we have metadata in a yaml file, no need to go to
                # Google Drive to get it. Just load from yaml
                self.metadata = self.load_metadata_from_yaml()

        self.latest_test_params = self._load_analysis_params()

    def test_parameters(self,
                        images_ndarray:  np.ndarray,
                        roi: Tuple['xmin', 'ymin', 'xmax', 'ymax'],
                        gaussian_width: int,
                        particle_z_diameter: int,
                        particle_xy_diameter: int,
                        brightness_percentile: int,
                        min_particle_mass: int,
                        bottom_slice: int,
                        top_slice: int,
                        time_point: int) -> dict:
        """Tests a set of particle finding parameters on one stack

        Runs the particle finding algorithm on only one stack to test a set
        of parameters. Much faster than finding particles in the whole trial.
        Useful if one stack in the trial is messing up linking of all the
        other stacks.

        Args:
            images_ndarray: Numpy array with all the image data
            roi: Tuple with xmin, ymin, xmax, ymax defining region of interest
            gaussian_width: int specifying the width of Gaussian blur kernel
            particle_z_diameter: int for maximum particle size in z
            particle_xy_diameter: int for maximum particle size in x and y
            brightness_percentile: int for brightness threshold as percentile
            min_particle_mass: int for minimum integrated mass cutoff
            bottom_slice: int for which slice to use as bottom of the stack
            top_slice: int for which slice to use as top of the stack
            time_point: int for which timepoint to test parameters on


        Returns:
            mito_candidates: dict of info about potential mitochondria
        """

        analysisParams = dict(
                {'gaussian_width': gaussian_width,
                 'particle_z_diameter': particle_z_diameter,
                 'particle_xy_diameter': particle_xy_diameter,
                 'brightness_percentile': brightness_percentile,
                 'min_particle_mass': min_particle_mass,
                 'bottom_slice': bottom_slice,
                 'top_slice': top_slice,
                 'roi': roi})
        particle_diameter = (particle_z_diameter,
                             particle_xy_diameter,
                             particle_xy_diameter)

        with open(self.param_test_history_file, 'a') as output_file:
            yaml.dump(analysisParams, output_file, explicit_start=True)
            # dump the latest analysis into the history file

        slices_to_analyze = self.image_array[time_point,
                                             bottom_slice:top_slice,
                                             roi[1]:roi[3],
                                             roi[0]:roi[2]]

        print('Looking for mitochondria...')

        self.mito_candidates = tp.locate(slices_to_analyze,
                                         particle_diameter,
                                         percentile=brightness_percentile,
                                         minmass=min_particle_mass,
                                         noise_size=gaussian_width,
                                         characterize=True)

        # Correct for roi offset
        self.mito_candidates['x'] = (self.mito_candidates['x'] +
                                     min([roi[0], roi[2]]))
        self.mito_candidates['y'] = (self.mito_candidates['y'] +
                                     min([roi[1], roi[3]]))

        return self.mito_candidates

    def run_batch(self,
                  images_ndarray:  np.ndarray,
                  roi: Tuple['xmin', 'ymin', 'xmax', 'ymax'],
                  gaussian_width: int,
                  particle_z_diameter: int,
                  particle_xy_diameter: int,
                  brightness_percentile: int,
                  min_particle_mass: int,
                  bottom_slice: int,
                  top_slice: int,
                  tracking_seach_radius: int,
                  last_timepoint: int,
                  notes: str) -> None:
        """Runs particle finding algorithm on entire trial.

        Runs the particle finding algorithm on the whole trial. Rather than
        returning a dict with the particles, it just saves them directly to
        the trackpyBatchResults.yaml file in analyzed_data_location. Also
        saves unlinked particle finding results and images of results.

        Args:
            images_ndarray: Numpy array with all the image data
            roi: Tuple with xmin, ymin, xmax, ymax defining region of interest
            gaussian_width: int specifying the width of Gaussian blur kernel
            particle_z_diameter: int for maximum particle size in z
            particle_xy_diameter: int for maximum particle size in x and y
            brightness_percentile: int for brightness threshold as percentile
            min_particle_mass: int for minimum integrated mass cutoff
            bottom_slice: int for which slice to use as bottom of the stack
            top_slice: int for which slice to use as top of the stack
            tracking_seach_radius: int for maximum search radius for linking
            last_timepoint: int for last timepoint to analyze
            notes: str for short note about goal of this run


        Returns:
            None
        """

        slices_to_analyze = images_ndarray[:last_timepoint,
                                           bottom_slice:top_slice,
                                           roi[1]:roi[3],
                                           roi[0]:roi[2]]
        particle_diameter = (particle_z_diameter,
                             particle_xy_diameter,
                             particle_xy_diameter)
        save_location = self.analyzed_data_location
        metadata_save_location = str(
                save_location.joinpath('trackpyBatchParams.yaml'))

        # run batch of images with the current set of parameters
        self.mitos_from_batch = tp.batch(
                frames=slices_to_analyze,
                diameter=particle_diameter,
                percentile=brightness_percentile,
                minmass=min_particle_mass,
                noise_size=gaussian_width,
                meta=metadata_save_location,
                characterize=True)

        # link the particles we found between time points
        linked = tp.link_df(self.mitos_from_batch,
                            tracking_seach_radius,
                            pos_columns=['x', 'y', 'z'])

        # only keep trajectories where point appears in all frames
        self.linked_mitos = tp.filter_stubs(
                linked, last_timepoint)

        # Correct for roi offset
        self.mitos_from_batch['x'] = (self.mitos_from_batch['x'] +
                                      min([roi[0], roi[2]]))
        self.mitos_from_batch['y'] = (self.mitos_from_batch['y'] +
                                      min([roi[1], roi[3]]))
        self.linked_mitos['x'] = (self.linked_mitos['x'] +
                                  min([roi[0], roi[2]]))
        self.linked_mitos['y'] = (self.linked_mitos['y'] +
                                  min([roi[1], roi[3]]))

        # add other parameters to the yaml created by tp.batch
        if notes == 'Notes for analysis run':
            notes = 'none'
        other_param_dict = dict(
                {'tracking_seach_radius': tracking_seach_radius,
                 'bottom_slice': bottom_slice,
                 'top_slice': top_slice,
                 'last_timepoint': last_timepoint,
                 'roi': roi,
                 'notes': notes})
        with open(save_location.joinpath('trackpyBatchParams.yaml'),
                  'a') as yamlfile:
            yaml.dump(other_param_dict, yamlfile, default_flow_style=False)

        # load the file again now that it has all parameters
        with open(save_location.joinpath('trackpyBatchParams.yaml'),
                  'r') as yamlfile:
            cur_yaml = yaml.load(yamlfile)

        # dump the latest analysis into the history file
        with open(self.batch_history_file, 'a') as yamlfile:
            yaml.dump(cur_yaml, yamlfile, explicit_start=True)

        # save the results to a yaml file
        linked_mitos_dict = self.linked_mitos.reset_index(
                drop=True).to_dict(orient='index')
        with open(save_location.joinpath('trackpyBatchResults.yaml'),
                  'w') as yamlfile:
            yaml.dump(linked_mitos_dict, yamlfile,
                      explicit_start=True, default_flow_style=False)
        mitos_from_batch_dict = self.mitos_from_batch.reset_index(
                drop=True).to_dict(orient='index')
        with open(save_location.joinpath('unlinkedTrackpyBatchResults.yaml'),
                  'w') as yamlfile:
            yaml.dump(mitos_from_batch_dict, yamlfile,
                      explicit_start=True, default_flow_style=False)

        self.save_diag_figs(images_ndarray, self.linked_mitos,
                            self.mitos_from_batch, save_location)

    def link_mitos(self,
                   tracking_seach_radius: int,
                   last_timepoint: int) -> None:
        """Links previously found mitochondria into trajectories

        Runs the particle linking algorithm using existing particle locations.
        Much faster than finding particles again in the whole trial and then
        linking with different parameters. Useful if particle locations look
        good, but the trajectories look wrong. Rather than returning a dict
        with the particles, it just saves them directly to the
        trackpyBatchResults.yaml file in analyzed_data_location. Also
        saves unlinked particle finding results and images of results.

        Args:
            tracking_seach_radius: int for maximum search radius for linking
            last_timepoint: int for last timepoint to analyze


        Returns:
            None
        """

        print('linking partiles...')

        save_location = self.analyzed_data_location

        # link the particles we found between time points
        linked = tp.link_df(
                self.mitos_from_batch.loc[
                        (self.mitos_from_batch['frame'] < last_timepoint)],
                search_range=tracking_seach_radius,
                pos_columns=['z', 'y', 'x'])

        # only keep trajectories where point appears in all frames
        self.linked_mitos = tp.filter_stubs(linked, last_timepoint)

        # load the file of parameters that tp.batch saved previously
        with open(save_location.joinpath('trackpyBatchParams.yaml'),
                  'r') as yamlfile:
            old_yaml = yaml.load(yamlfile)

        # add parameters used for linking to the yaml created by tp.batch
        other_param_dict = dict(
                {'tracking_seach_radius': tracking_seach_radius,
                 'last_timepoint': last_timepoint})
        new_param_dict = {**old_yaml, **other_param_dict}
        with open(save_location.joinpath('trackpyBatchParams.yaml'),
                  'w') as yamlfile:
            yaml.dump(new_param_dict, yamlfile, default_flow_style=False)

        # dump the latest analysis into the history file
        with open(self.batch_history_file, 'a') as yamlfile:
            yaml.dump(new_param_dict, yamlfile, explicit_start=True)

        # save the results to a yaml file
        linked_mitos_dict = self.linked_mitos.reset_index(
                drop=True).to_dict(orient='index')
        with open(save_location.joinpath('trackpyBatchResults.yaml'),
                  'w') as yamlfile:
            yaml.dump(linked_mitos_dict, yamlfile,
                      explicit_start=True, default_flow_style=False)

    def calculate_strain(self):
        """Calculates strain in the TRN using mitochondria positions

        Uses adjacent pairs of mitochondria as fiducial markers to calculate
        strain as a function of y-coordinate and time. Saves the results to
        strain.yaml file in analyzed_data_location.

        Args:
            None

        Returns:
            None
        """
        mitos_df = self.linked_mitos.copy(deep=True)
        mito_locations = mitos_df.loc[:, ['frame', 'particle', 'x', 'y', 'z']]

        def get_pressure(row):
            frame_num = int(row['frame'])
            pressure = self.metadata['pressure_kPa'][frame_num]
            return pressure

        mito_locations['pressure'] = mito_locations.apply(
                lambda row: get_pressure(row), axis=1)

        def get_rest_location(particle, axis):

            x_rest = mito_locations.loc[
                    (mito_locations['particle'] == particle) &
                    (mito_locations['pressure'] == 0)]['x'].mean()
            y_rest = mito_locations.loc[
                    (mito_locations['particle'] == particle) &
                    (mito_locations['pressure'] == 0)]['y'].mean()
            z_rest = mito_locations.loc[
                    (mito_locations['particle'] == particle) &
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

        dict_to_save = mito_locations.to_dict('list')

        with open(self.analyzed_data_location.joinpath('mito_locations.yaml'),
                  'w') as yamlfile:
            yaml.dump(dict_to_save, yamlfile,
                      explicit_start=True, default_flow_style=False)

        mito_pairs = pd.DataFrame(columns=['x_dist', 'y_dist', 'z_dist',
                                           'x_1', 'x_2', 'y_1', 'y_2',
                                           'z_1', 'z_2', 'total_dist'
                                           'particle_1', 'particle_2', 'frame',
                                           'pair_id'])
        mito_pairs_dicts = []
        for frame in mito_locations.frame.unique():
            this_frame = mito_locations.loc[
                    (mito_locations['frame'] == frame)].copy()
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

                total_dist = distance.euclidean([x_1, y_1, z_1],
                                                [x_2, y_2, z_2])

                particle_1 = int(this_frame.iloc[this_particle]['particle'])
                particle_2 = int(
                        this_frame.iloc[this_particle + 1]['particle'])

                mito_pairs_dicts.append({'x_dist': x_dist, 'x_1': x_1,
                                         'x_2': x_2,
                                         'y_dist': y_dist, 'y_1': y_1,
                                         'y_2': y_2,
                                         'z_dist': z_dist, 'z_1': z_1,
                                         'z_2': z_2,
                                         'particle_1': particle_1,
                                         'particle_2': particle_2,
                                         'total_dist': total_dist,
                                         'frame': frame,
                                         'pair_id': this_particle})
        mito_pairs = pd.DataFrame(mito_pairs_dicts)
        mito_pairs['pressure'] = mito_pairs.apply(
                lambda row: get_pressure(row), axis=1)

        def get_rest_distance(pair, axis):
            euclidean_rest_dist = mito_pairs.loc[
                    (mito_pairs['pair_id'] == pair) &
                    (mito_pairs['pressure'] == 0)]['total_dist'].mean()
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
                lambda pair: get_rest_distance(pair['pair_id'], 'euclid'),
                axis=1)
        mito_pairs['x_rest_dist'] = mito_pairs.apply(
                lambda pair: get_rest_distance(pair['pair_id'], 'x'), axis=1)
        mito_pairs['y_rest_dist'] = mito_pairs.apply(
                lambda pair: get_rest_distance(pair['pair_id'], 'y'), axis=1)
        mito_pairs['z_rest_dist'] = mito_pairs.apply(
                lambda pair: get_rest_distance(pair['pair_id'], 'z'), axis=1)

        mito_pairs['strain'] = (
                (mito_pairs['total_dist'] - mito_pairs['rest_dist'])
                / mito_pairs['rest_dist'])
        mito_pairs['x_strain'] = (
                (mito_pairs['x_dist'] - mito_pairs['x_rest_dist'])
                / mito_pairs['x_rest_dist'])
        mito_pairs['y_strain'] = (
                (mito_pairs['y_dist'] - mito_pairs['y_rest_dist'])
                / mito_pairs['y_rest_dist'])
        mito_pairs['z_strain'] = (
                (mito_pairs['z_dist'] - mito_pairs['z_rest_dist'])
                / mito_pairs['z_rest_dist'])
        self.strain = np.empty((mito_pairs['frame'].nunique(),
                               mito_pairs['pair_id'].nunique()))
        self.ycoords_for_strain = np.empty((mito_pairs['frame'].nunique(),
                                            mito_pairs['pair_id'].nunique()))
        for stack in range(mito_pairs['frame'].nunique()):
            for interval in range(mito_pairs['pair_id'].nunique()):
                self.strain[stack, interval] = mito_pairs.loc[
                        (mito_pairs['frame'] == stack) &
                        (mito_pairs['pair_id'] == interval)]['strain']
                self.ycoords_for_strain[stack, interval] = mito_pairs.loc[
                        (mito_pairs['frame'] == stack) &
                        (mito_pairs['pair_id'] == interval)]['y_1']

        # save results
        df_to_save = mito_pairs[['frame', 'pair_id', 'pressure',
                                 'particle_1', 'particle_2', 'total_dist',
                                 'x_dist', 'y_dist', 'z_dist',
                                 'rest_dist', 'x_rest_dist', 'y_rest_dist',
                                 'z_rest_dist', 'strain', 'x_strain',
                                 'y_strain', 'z_strain']]
        dict_to_save = df_to_save.to_dict('list')

        with open(self.analyzed_data_location.joinpath('strain.yaml'),
                  'w') as yamlfile:
            yaml.dump(dict_to_save, yamlfile,
                      explicit_start=True, default_flow_style=False)

    def save_diag_figs(self,
                       image_array: np.ndarray,
                       linked_mitos: pd.DataFrame,
                       mitos_from_batch: pd.DataFrame,
                       save_location: pathlib.Path):
        """Saves figs with trajectories and individual particle candidates

        Puts the particles that were found on a max projection of each stack
        and saves the image. Makes it faster to load these figs later compared
        to the time it takes to load the raw data.

        Args:
            image_array: Numpy array with the raw image data
            linked_mitos: Dataframe with the linked particle finding results
            mitos_from_batch: Dataframe with unlinked particle finding results
            save_location: pathlib.Path to where the images are saved

        Returns:
            None
        """
        traj_fig_num = 13
        one_stack_fig_num = 31

        # Create directory if necessary
        if not save_location.joinpath('diag_images/').is_dir():
            save_location.joinpath('diag_images/').mkdir()

        # Save image with trajectories
        image_to_display = np.amax(image_array[0], 0)  # collapse z axis
        image_to_display = image_to_display.squeeze()

        trajectory_fig = plt.figure(figsize=(8.5, 8.5*2),
                                    frameon=False, num=traj_fig_num)
        trajectory_ax = plt.Axes(trajectory_fig, [0., 0., 1., 1.])
        trajectory_fig.set_label('trajectories')
        trajectory_fig.tight_layout(pad=0)
        trajectory_ax.margins(0, 0)
        trajectory_ax.set_axis_off()
        trajectory_fig.add_axes(trajectory_ax)
        trajectory_ax.imshow(image_to_display,
                             vmin=int(np.amin(image_to_display)),
                             vmax=200)  # set max pixel to 200 for visibility

        try:
            theCount = 0  # ah ah ah ah
            for i in range(max(linked_mitos['particle'])):
                this_particle = linked_mitos.loc[
                        linked_mitos['particle'] == i+1]
                if not this_particle.empty:
                    theCount += 1
                    this_particle.plot(
                            x='x',
                            y='y',
                            ax=trajectory_ax,
                            color='#FB8072',
                            marker='None',
                            linestyle='-')
                    trajectory_ax.text(
                            this_particle['x'].mean() + 15,
                            this_particle['y'].mean(),
                            str(int(this_particle.iloc[0][
                                     'particle'])), color='white')
                    trajectory_ax.legend_.remove()

            trajectory_fig.savefig(save_location.joinpath(
                     'diag_images/trajectory_fig.png'),
                    dpi=72, pad_inches=0)
            plt.close(trajectory_fig)
        except ValueError:
            if len(linked_mitos) == 0:
                warnings.warn('No particles were found at all '
                              'time points. Try expanding search '
                              'radius or changing particle finding '
                              'parameters.')
            else:
                raise
        except TypeError:
            if linked_mitos is None:
                pass
            else:
                raise
        finally:
            if plt.fignum_exists(traj_fig_num):
                plt.close(trajectory_fig)

        try:
            # Save individual images with all particles, linked mitos labeled
            for stack in range(max(mitos_from_batch['frame'])):
                image_to_display = np.amax(image_array[stack], 0)  # collapse z
                image_to_display = image_to_display.squeeze()

                one_stack_fig = plt.figure(figsize=(8.5, 8.5*2),
                                           frameon=False, num=traj_fig_num)
                one_stack_ax = plt.Axes(one_stack_fig, [0., 0., 1., 1.])
                one_stack_fig.set_label('trajectories')
                one_stack_fig.tight_layout(pad=0)
                one_stack_ax.margins(0, 0)
                one_stack_ax.set_axis_off()
                one_stack_fig.add_axes(one_stack_ax)
                one_stack_ax.imshow(image_to_display,
                                    vmin=int(np.amin(image_to_display)),
                                    vmax=200)  # set max pixel to 200

                df_for_plot = mitos_from_batch.loc[
                            mitos_from_batch[
                                    'frame'] == stack]
                df_for_plot.plot(x='x', y='y', ax=one_stack_ax,
                                 color='#FB8072', marker='o', linestyle='None')
                theCount = 0  # ah ah ah ah
                if len(linked_mitos) != 0:
                    for i in range(max(linked_mitos['particle'])):
                        this_particle = linked_mitos.loc[
                                linked_mitos['particle'] == i+1]
                        if not this_particle.empty:
                            theCount += 1
                            one_stack_ax.text(
                                    this_particle['x'].mean() + 15,
                                    this_particle['y'].mean(),
                                    str(int(this_particle.iloc[0][
                                             'particle'])), color='white')
                    one_stack_ax.legend_.remove()
                one_stack_fig.savefig(save_location.joinpath(
                 ('diag_images/stack_' + str(stack) + '_fig.png')),
                    dpi=72, pad_inches=0)
                plt.close(one_stack_fig)
        except ValueError:
            if len(mitos_from_batch) == 0:
                warnings.warn('No particles were found. Try changing particle'
                              ' finding parameters.')
            else:
                raise
        except TypeError:
            if linked_mitos is None:
                pass
            else:
                raise
        finally:
            if plt.fignum_exists(traj_fig_num):
                plt.close(trajectory_fig)
            if plt.fignum_exists(one_stack_fig_num):
                plt.close(one_stack_fig)

    def _load_images_from_disk(self) -> Tuple[np.array, ND2Reader]:
        """Accesses the image data from the file."""
        images = pims.open(self.filename)
        try:
            images.bundle_axes = ['z', 'y', 'x']
            image_array = np.asarray(images)
            image_array = image_array.squeeze()
        except ValueError:
            if self.filename[-6:-4] == 'bf':
                # This is a brightfield image to locate actuator
                image_array = np.asarray(images)
                image_array = image_array.squeeze()
            else:
                raise

        return image_array, images

    def load_metadata_from_yaml(self) -> dict:
        """Loads metadata from an existing yaml file.

        Uses the instance attribute self.metadata_file_path to load saved
        metadata and hold it as instance attribute self.metadata

        Args:
            None

        Returns:
            metadata: dict containing the metadata for this trial
        """
        with open(self.metadata_file_path, 'r') as yamlfile:
            metadata = yaml.load(yamlfile)

        return metadata

    def get_analysis_status(self) -> str:
        """Returns analysis status for this trial."""
        analysis_status = self.metadata['analysis_status']

        return analysis_status

    def _load_analysis_params(self) -> dict:
        """Loads analysis parameters from an existing yaml file.

        Looks at the batch history file and loads up the last set of
        parameters used in an analysis.

        Args:
            None

        Returns:
            all_params: dict containing last set of analysis parameters
        """
#        try:
#            with open(self.param_test_history_file, 'r') as yamlfile:
#                entire_history = yaml.load_all(yamlfile)
#                trackpy_locate_params = None
#                for trackpy_locate_params in entire_history:  # get newest
#                    pass
#        except FileNotFoundError:
#            print('Previous parameter file not found. Using defaults.')
#            trackpy_locate_params = self.default_test_params
#            trackpy_locate_params['top_slice'] = self.metadata['stack_height']

        try:
            with open(self.batch_history_file, 'r') as yamlfile:
                entire_history = yaml.load_all(yamlfile)
                trackpy_batch_params = None
                for trackpy_batch_params in entire_history:  # get most recent
                    pass
        except FileNotFoundError:
            print('Previous batch parameter file not found. Using defaults.')
            trackpy_batch_params = self.default_test_params
            trackpy_batch_params['top_slice'] = self.metadata['stack_height']

#        all_params = {**trackpy_locate_params, **trackpy_batch_params}
        all_params = trackpy_batch_params
        if 'roi' not in all_params:
            try:
                all_params['roi'] = [0,
                                     0,
                                     self.image_array.shape[3],
                                     self.image_array.shape[2]]
            except NameError:
                all_params['roi'] = [0,
                                     0,
                                     1200,
                                     600]

        return all_params

    def write_metadata_to_yaml(self, metadata_dict: dict):
        """Takes metadata dict and writes it to a yaml file"""
        with open(self.metadata_file_path, 'w') as output_file:
                yaml.dump(metadata_dict, output_file,  # create file
                          explicit_start=True, default_flow_style=False)

    def _retrieve_metadata(self, images: ND2Reader) -> dict:
        """Retrieves metadata from Google Drive and from the image file.

        Goes to google drive and retrieves the metadata for the current trial
        and then combines it with some useful metadata from the image file
        itself. Can be slow and sometimes hits Google API limit when running
        a script that has to get some metadata from all trials.

        Args:
            images (ND2Reader): Reader object containing images for this trial

        Returns:
            metadata_dict: dict containing all metadata for this trial
        """
        # Access google drive spreadsheet
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        keyfile = 'SenseOfTouchResearch-e5927f56c4d0.json'
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
                keyfile, scope)
        gspread_client = gspread.authorize(credentials)
        metadata_spread = gspread_client.open_by_key(
                '1LsTdPBOW79XSkk5DJv2ckiVTvpofAOOd_dJL4cgtxBQ')
        metadata_worksheet = metadata_spread.worksheet("MetadataResponses")
        current_id_cell = metadata_worksheet.find(self.experiment_id)
        row_of_metadata = metadata_worksheet.row_values(current_id_cell.row)
        row_of_keys = metadata_worksheet.row_values(1)
        gdrive_metadata_dict = dict(zip(row_of_keys, row_of_metadata))

        pressure_kPa = []
        for key in gdrive_metadata_dict.keys():
            if key[0:23] == 'Actuator pressure (kPa)':
                split1 = key.split(' ')
                split2 = split1[-1].split(']')
                stack_num = int(split2[0]) - 1
                pressure_kPa.insert(int(stack_num),
                                    int(gdrive_metadata_dict[key]))
        gdrive_metadata_dict['pressure_kPa'] = pressure_kPa

        # Access the metadata from the file
        meta = images.metadata
        keys_to_keep = ['height', 'width',
                        'date', 'total_images_per_channel', 'z_levels',
                        'channels', 'pixel_microns', 'num_frames']
        metadata_from_scope = {key: (meta[key]) for key in keys_to_keep}

        # Combine metadata from both sources into one dictionary
        combined_metadata = {**gdrive_metadata_dict, **metadata_from_scope}

        # Do a bit of processing to make things clearer
        time_str = combined_metadata['Timestamp']
        bleach_date = combined_metadata['Bleach Date']
        bleach_time = combined_metadata['Bleach Time']
        if 'Trial_rating' not in combined_metadata:
            combined_metadata['Trial rating'] = 'No rating given'

        metadata_dict = {
                'Experiment_id': self.experiment_id,
                'slice_height_pix': int(combined_metadata['height']),
                'slice_width_pix': int(combined_metadata['width']),
                'stack_height': int(
                        combined_metadata['z_levels'][-1] + 1),
                'num_timepoints': int(
                        combined_metadata['num_frames']),
                'timestamp': datetime.datetime.strptime(
                        time_str, '%m/%d/%Y %H:%M:%S'),
                'total_images': int(combined_metadata[
                        'total_images_per_channel']),
                'microscope_channel': combined_metadata['channels'][0],
                'pixel_microns': float(combined_metadata[
                        'pixel_microns']),
                'bleach_time': datetime.datetime.strptime(
                        bleach_date + ' ' + bleach_time,
                        '%m/%d/%Y %H:%M:%S %p'),
                'cultivation_temp': int(combined_metadata[
                        'Cultivation Temperature (°C)']),
                'device_ID': combined_metadata['Device ID'],
                'user': combined_metadata['Email Address'],
                'worm_life_stage': combined_metadata['Life stage'],
                'microscope': combined_metadata['Microscope'],
                'neuron': combined_metadata['Neuron'],
                'notes': combined_metadata['Notes'],
                'num_eggs': int(combined_metadata[
                        'Number of eggs visible']),
                'microscope_objective': combined_metadata[
                        'Objective'],
                'purpose': combined_metadata['Purpose of Experiment'],
                'worm_strain': combined_metadata['Worm Strain'],
                'head_orientation': combined_metadata[
                        'Worm head orientation'],
                'vulva_orientation': combined_metadata[
                        'Worm vulva orientation'],
                'trial_rating': combined_metadata['Trial rating'],
                'pressure_kPa': combined_metadata['pressure_kPa']}

        return metadata_dict


if __name__ == '__main__':
    import ssn_image_analysis_gui_controller
    controller = ssn_image_analysis_gui_controller.StrainGUIController()
    controller.run()
