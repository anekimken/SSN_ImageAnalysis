#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:24:53 2019

@author: Adam Nekimken
"""
# TODO: UPDATE DOCSTRINGS!!!!
# import sys
import os
import pathlib
import datetime
import time
import warnings
from typing import Tuple, Any

import numpy as np
import pims
import pandas as pd
from nd2reader import ND2Reader
# from scipy import stats
from scipy import spatial
import yaml

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
    using particle-finding algorithm from Crocker–Grier

    Args:
        filename: A string indicating where the image file is located
        load_images: A bool indicating if images should be loaded on init
        overwrite_metadata: A bool indicating to overwrite yaml metadata file

    Attributes:
        filename: A string indicating where the image file is located
        load_images: A bool indicating if images should be loaded on init
        overwrite_metadata: A bool indicating to overwrite yaml metadata file
        experiment_id: A string identifying this particular trial
        analyzed_data_location: A string with the location of analyzed data
        metadata_file_path: A string with absolute path to metadata.yaml
        metadata: A dictionary containing all the metadata
        image_array: Numpy array with image data. Can be null if not loaded

    """
    STATUSES = ['No metadata.yaml file',
                'No analysis status yet',
                'Not started',
                'Testing parameters',
                'Testing parameters for batch',
                'Strain calculated',
                'Failed - too close to edge',
                'Failed - not bright enough']

    def __init__(self):
        self.mito_candidates = None
        self.linked_mitos = None
        self.latest_test_params = None
        self.default_test_params = {'gaussian_width': 3,
                                    'particle_z_diameter': 21,
                                    'particle_xy_diameter': 15,
                                    'brightness_percentile': 50,
                                    'min_particle_mass': 200,
                                    'bottom_slice': 0,
                                    'top_slice': 2,
                                    'time_point': 1,
                                    'tracking_seach_radius': 20,
                                    'last_timepoint': 11}

    def load_trial(self,
                   filename: str,
                   load_images: bool = True,
                   overwrite_metadata: bool = False) -> Tuple[np.array, dict]:
        self.filename = filename
        self.load_images = load_images
        self.overwrite_metadata = overwrite_metadata

        """Loads the data and metadata for this trial from disk.

        Loads the data needed for analyzing the trial. Also handles
        reading/writing of the metadata yaml file. First checks if the only
        thing that needs to be loaded is the metadata, since this is fastest.
        Otherwise, loads the image file, then either writes a new metadata
        file with data from the image file and Google Drive, or loads the
        existing metadata from a yaml file. It takes a long time to load a
        large amount of data, so the goal here is to load as little data
        as possible.

        Returns:
            image_array (np.array): Image data. Can be empty if not loaded
            metadata (dict): All the metadata

        """
        # Get the experiment ID from the filename and load the data
        basename = os.path.basename(self.filename)
        self.experiment_id = os.path.splitext(basename)[0]
        self.analyzed_data_location = pathlib.Path(
                '/Users/adam/Documents/SenseOfTouchResearch/'
                'SSN_ImageAnalysis/AnalyzedData/' + self.experiment_id + '/')
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

        # Create directory if necessary
        if not self.analyzed_data_location.is_dir():
            self.analyzed_data_location.mkdir()

        # TODO: Load other analyzed data here too
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

        start_time = time.time()
        if (self.load_images is False and  # don't load images
                self.overwrite_metadata is False and  # don't overwrite
                self.metadata_file_path.is_file() is True):  # yaml exists
            # only load metadata in this case
            self.metadata = self.load_metadata_from_yaml()
            # TODO: create empty numpy array of the right size
            self.image_array = np.array([2, 2, 2, 2])

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

        finish_time = time.time()
        local_image = self.image_array
        local_self = self
        print('Loaded file in ' + str(round(finish_time - start_time)) +
              ' seconds.')

    def test_parameters(self,
                        gaussian_width: int,
                        particle_z_diameter: int,
                        particle_xy_diameter: int,
                        brightness_percentile: int,
                        min_particle_mass: int,
                        bottom_slice: int,
                        top_slice: int,
                        time_point: int) -> dict:
        # TODO: docstring
        # TODO: update formatting of names here and in saved files

        analysisParams = dict(
                {'gaussian_width': gaussian_width,
                 'particle_z_diameter': particle_z_diameter,
                 'particle_xy_diameter': particle_xy_diameter,
                 'brightness_percentile': brightness_percentile,
                 'min_particle_mass': min_particle_mass,
                 'bottom_slice': bottom_slice,
                 'top_slice': top_slice})
        particle_diameter = (particle_z_diameter,
                             particle_xy_diameter,
                             particle_xy_diameter)

        with open(self.param_test_history_file, 'a') as output_file:
            yaml.dump(analysisParams, output_file, explicit_start=True)
            # dump the latest analysis into the history file

        slicesToAnalyze = self.image_array[time_point, bottom_slice:top_slice]
    #     for i in range(0,imageCurrent.shape[0]):
    #         maxProj = dispMaxProjection(
    # slicesToAnalyze[i])#,metadataCurrent,timePoint)

        # MAYBE: make this a new thread?
        print('Looking for mitochondria...')
        start_time = time.time()
        self.mito_candidates = tp.locate(slicesToAnalyze,
                                         particle_diameter,
                                         percentile=brightness_percentile,
                                         minmass=min_particle_mass,
                                         noise_size=gaussian_width,
                                         characterize=True)
        finish_time = time.time()
        print('Time to test parameters for locating mitochondria was ' +
              str(round(finish_time - start_time)) + ' seconds.')

        return self.mito_candidates

    def run_batch(self,
                  images_ndarray:  Any,
                  gaussian_width: int,
                  particle_z_diameter: int,
                  particle_xy_diameter: int,
                  brightness_percentile: int,
                  min_particle_mass: int,
                  bottom_slice: int,
                  top_slice: int,
                  tracking_seach_radius: int,
                  last_timepoint: int) -> dict:

        slices_to_analyze = images_ndarray[:last_timepoint,
                                           bottom_slice:top_slice]
        particle_diameter = (particle_z_diameter,
                             particle_xy_diameter,
                             particle_xy_diameter)
        save_location = self.analyzed_data_location
        metadata_save_location = str(
                save_location.joinpath('trackpyBatchParams.yaml'))

        start_time = time.time()
        # run batch of images with the current set of parameters
        self.mitos_from_batch = tp.batch(
                frames=slices_to_analyze,
                diameter=particle_diameter,
                percentile=brightness_percentile,
                minmass=min_particle_mass,
                noise_size=gaussian_width,
                meta=metadata_save_location,
                characterize=True)
        batch_done_time = time.time()

        # link the particles we found between time points
        linked = tp.link_df(self.mitos_from_batch,
                            tracking_seach_radius,
                            pos_columns=['x', 'y', 'z'])

        # only keep trajectories where point appears in all frames
        self.linked_mitos = tp.filter_stubs(
                linked, last_timepoint)
        link_done_time = time.time()

        # add other parameters to the yaml created by tp.batch
        other_param_dict = dict(
                {'tracking_seach_radius': tracking_seach_radius,
                 'bottom_slice': bottom_slice,
                 'top_slice': top_slice,
                 'last_timepoint': last_timepoint})
        with open(save_location.joinpath('trackpyBatchParams.yaml'),
                  'a') as yamlfile:
            yaml.dump(other_param_dict, yamlfile, default_flow_style=False)

        # load the file of parameters that tp.batch just saved
        with open(save_location.joinpath('trackpyBatchParams.yaml'),
                  'r') as yamlfile:
            cur_yaml = yaml.load(yamlfile)  # , Loader=PrettySafeLoader)
            # Note the PrettySafeLoader which can load tuples

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

            # TODO: save all particles found, can be linked later

        print('Done running file. Batch find took ' +
              str(round(batch_done_time - start_time)) + ' seconds. ' +
              'Linking and filtering took ' +
              str(round(link_done_time - batch_done_time)) + ' seconds.')

        # return self.linked_mitos

    def link_mitos(self,
                   tracking_seach_radius: int,
                   last_timepoint: int):
        """Links previously found mitochondria into trajectories"""

        print('linking partiles...')

        save_location = self.analyzed_data_location
        start_time = time.time()

        # link the particles we found between time points
        linked = tp.link_df(self.mitos_from_batch,
                            tracking_seach_radius,
                            pos_columns=['x', 'y', 'z'])

        # only keep trajectories where point appears in all frames
        self.linked_mitos = tp.filter_stubs(linked, last_timepoint)
        link_done_time = time.time()

        # load the file of parameters that tp.batch saved previously
        with open(save_location.joinpath('trackpyBatchParams.yaml'),
                  'r') as yamlfile:
            old_yaml = yaml.load(yamlfile)

        # add other parameters to the yaml created by tp.batch
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

        print('Done linking trial file. Linking and filtering took ' +
              str(round(link_done_time - start_time)) + ' seconds.')

    def calculate_strain(self):
        """Calculates strain in the TRN using mitochondria positions"""
        print('Calculating strain!')
        mitos_df = self.linked_mitos.copy(deep=True)
        num_trajectories = mitos_df['particle'].nunique()
        num_frames = mitos_df['frame'].nunique()
        distances = np.empty([num_frames, num_trajectories - 1])
        for stack in range(num_frames):
            # sort mitochondria in this stack by y values
            current_stack = mitos_df.loc[mitos_df['frame'] == stack]
            sorted_stack = current_stack.sort_values(['y'])
            sorted_stack.reset_index(inplace=True, drop=True)
            for particle in range(num_trajectories - 1):
                # calculate pairwise distances
                mito1 = sorted_stack.loc[particle, ['x', 'y', 'z']].values
                mito2 = sorted_stack.loc[particle + 1, ['x', 'y', 'z']].values
                distances[stack, particle] = spatial.distance.euclidean(
                        mito1, mito2)

        # calculate change in distance over time
        self.strain = (distances - distances[0])/distances[0]

        # save results
        with open(self.analyzed_data_location.joinpath('strain.yaml'),
                  'w') as yamlfile:
            yaml.dump(self.strain, yamlfile,
                      explicit_start=True, default_flow_style=False)

    def _load_images_from_disk(self) -> Tuple[np.array, ND2Reader]:
        """Accesses the image data from the file."""
        images = pims.open(self.filename)
        images.bundle_axes = ['z', 'y', 'x']
        image_array = np.asarray(images)
        image_array = image_array.squeeze()

        return image_array, images

    def load_metadata_from_yaml(self) -> dict:
        """Loads metadata from an existing yaml file."""
        with open(self.metadata_file_path, 'r') as yamlfile:
            metadata = yaml.load(yamlfile)

        return metadata

    def get_analysis_status(self) -> str:
        """Returns analysis status for this trial."""
        analysis_status = self.metadata['analysis_status']

        return analysis_status

    def _load_analysis_params(self) -> dict:
        """Loads analysis parameters from an existing yaml file."""
        try:
            with open(self.param_test_history_file, 'r') as yamlfile:
                entire_history = yaml.load_all(yamlfile)
                trackpy_locate_params = None
                for trackpy_locate_params in entire_history:  # get most recent
                    pass
        except FileNotFoundError:
            print('Previous parameter file not found. Using defaults.')
            trackpy_locate_params = self.default_test_params
            trackpy_locate_params['top_slice'] = self.metadata['stack_height']

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

        all_params = {**trackpy_locate_params, **trackpy_batch_params}

        return all_params

    def write_metadata_to_yaml(self, metadata_dict: dict):
        """Takes metadata dict and writes it to a yaml file"""
        with open(self.metadata_file_path, 'w') as output_file:
                yaml.dump(metadata_dict, output_file,  # create file
                          explicit_start=True, default_flow_style=False)

    def _retrieve_metadata(self, images: ND2Reader) -> dict:
        """Retrieves metadata from Google Drive and from the image file.

        Args:
            images (ND2Reader): Reader object containing images for this trial

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

        # Access the metadata from the file
        meta = images.metadata
        keys_to_keep = ['height', 'width',
                        'date', 'total_images_per_channel', 'z_levels',
                        'channels', 'pixel_microns', 'num_frames']
        metadata_from_scope = {key: (meta[key]) for key in keys_to_keep}

        # Combine metadata from both sources into one dictionary
        combined_metadata = {**gdrive_metadata_dict, **metadata_from_scope}

        time_str = combined_metadata['Timestamp']
        bleach_date = combined_metadata['Bleach Date']
        bleach_time = combined_metadata['Bleach Time']
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
                        'Worm vulva orientation']}

        return metadata_dict


class PrettySafeLoader(yaml.SafeLoader):  # not sure if I need this anymore
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


if __name__ == '__main__':
    import ssn_image_analysis_gui_controller
    controller = ssn_image_analysis_gui_controller.StrainGUIController()
    controller.run()
