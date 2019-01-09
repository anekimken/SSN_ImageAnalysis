#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:24:53 2019

@author: Adam Nekimken
"""
# TODO: DOCSTRINGS!!!!
import sys
import os
import pathlib
import datetime
import warnings

import numpy as np
import pandas as pd
import pims
from PIL import Image
from nd2reader import ND2Reader
from scipy import ndimage as ndi
from scipy import stats
import yaml

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore",
                        message=("Using a non-tuple sequence for "
                                 "multidimensional indexing is deprecated"))
warnings.filterwarnings("ignore", message="Reported average frame interval")


class StrainPropagationTrial:
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
    def __init__(self, filename, load_images=True, overwrite_metadata=False):
        self.filename = filename
        self.load_images = load_images
        self.overwrite_metadata = overwrite_metadata

        # Initialize yaml loader with tuple support
        # PrettySafeLoader.add_constructor(
        #         u'tag:yaml.org,2002:python/tuple',
        #         PrettySafeLoader.construct_python_tuple)

        # Get the experiment ID from the filename and load the data
        basename = os.path.basename(filename)
        self.experiment_id = os.path.splitext(basename)[0]
        self.analyzed_data_location = pathlib.Path(
                '/Users/adam/Documents/SenseOfTouchResearch/'
                'SSN_ImageAnalysis/AnalyzedData/' + self.experiment_id + '/')
        self.metadata_file_path = self.analyzed_data_location.joinpath(
                 'metadata.yaml')

        # Create directory if necessary
        if not self.analyzed_data_location.is_dir():
            self.analyzed_data_location.mkdir()

        self.load_trial()  # This function is what makes the init slow

    def load_trial(self):
        """Loads the data and metadata for this trial from disk.

        Loads the data needed for analyzing the trial. Also handles
        reading/writing of the metadata yaml file. First checks if the only
        thing that needs to be loaded is the metadata, since this is fastest.
        Otherwise, loads the image file, then either writes a new metadata
        file with data from the image file and Google Drive, or loads the
        existing metadata from a yaml file. It takes a long time to load a
        large amount of data, so the goal here is to load as little data
        as possible.

        """

        if (self.load_images is False and  # don't load images
                self.overwrite_metadata is False and  # don't overwrite
                self.metadata_file_path.is_file() is True):  # yaml exists
            # only load metadata in this condition
            self.metadata = self._load_metadata_from_yaml()
        else:
            self.image_array, images = self._load_images_from_disk()
            # If we don't have a yaml metadata file yet, we need to get
            # some information from Google Drive and build a yaml file
            if (self.metadata_file_path.is_file() is False or  # no yaml yet
                    self.overwrite_metadata is True):
                self.metadata = self._retrieve_metadata(images)
                self._write_metadata_to_yaml(self.metadata)
            else:
                # if we have metadata in a yaml file, no need to go to
                # Google Drive to get it. Just load from yaml
                self.metadata = self._load_metadata_from_yaml()

    def test_parameters(self):
            pass

    def run_batch(self):
            pass

    def _load_images_from_disk(self):
        """Accesses the image data from the file."""
        images = pims.open(self.filename)
        print(type(images))
        images.bundle_axes = ['z', 'y', 'x']
        image_array = np.asarray(images)
        image_array = image_array.squeeze()

        return image_array, images

    def _load_metadata_from_yaml(self):
        """Loads metadata from an existing yaml file."""
        with open(self.metadata_file_path, 'r') as yamlfile:
            metadata = yaml.load(yamlfile)

        return metadata

    def _write_metadata_to_yaml(self, metadata_dict):
        """Takes metadata dict and writes it to a yaml file"""
        with open(self.metadata_file_path, 'w') as output_file:
                yaml.dump(metadata_dict, output_file,  # create file
                          explicit_start=True, default_flow_style=False)

    def _retrieve_metadata(self, images):
        """Retrieves metadata from Google Drive and from the image file.

        Args:
            images (nd2reader object): Images for this trial

        """

        # Access google drive spreadsheet
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        keyfile = 'SenseOfTouchResearch-e5927f56c4d0.json'
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
                keyfile, scope)
        c = gspread.authorize(credentials)
        metadataSpread = c.open_by_key(
                '1LsTdPBOW79XSkk5DJv2ckiVTvpofAOOd_dJL4cgtxBQ')
        metadataFrame = pd.DataFrame(
                metadataSpread.sheet1.get_all_records())
        gdriveMetadata = metadataFrame.loc[metadataFrame[(
                'Experiment ID')] == self.experiment_id]

        # Access the metadata from the file
        meta = images.metadata
        keysToKeep = ['height', 'width',
                      'date', 'total_images_per_channel',
                      'channels', 'pixel_microns']
        metadataFromScope = {key: meta[key] for key in keysToKeep}
        metadataFromScope = pd.DataFrame(metadataFromScope)

        # Combine metadata from the two sources to get all of it together
        gdriveMetadata = gdriveMetadata.reset_index(drop=True)
        metadataFromScope = metadataFromScope.reset_index(drop=True)
        retrieved_metadata = metadataFromScope.join(gdriveMetadata)

        # Format metadata into a big dictionary for writing to yaml
        time_str = retrieved_metadata.iloc[0]['Timestamp']
        bleach_date = retrieved_metadata.iloc[0]['Bleach Date']
        bleach_time = retrieved_metadata.iloc[0]['Bleach Time']
        metadata_dict = {
                'experiment_id': self.experiment_id,
                'slice_height_pix': int(retrieved_metadata.iloc[0]['height']),
                'slice_width_pix': int(retrieved_metadata.iloc[0]['width']),
                'timestamp': datetime.datetime.strptime(
                        time_str, '%m/%d/%Y %H:%M:%S'),
                'total_images': int(retrieved_metadata.iloc[0][
                        'total_images_per_channel']),
                'microscope_channel': retrieved_metadata.iloc[0]['channels'],
                'pixel_microns': float(retrieved_metadata.iloc[0][
                        'pixel_microns']),
                'bleach_time': datetime.datetime.strptime(
                        bleach_date + ' ' + bleach_time,
                        '%m/%d/%Y %H:%M:%S %p'),
                'cultivation_temp': retrieved_metadata.iloc[0][
                        'Cultivation Temperature (°C)'],
                'device_ID': retrieved_metadata.iloc[0]['Device ID'],
                'user': retrieved_metadata.iloc[0]['Email Address'],
                'worm_life_stage': retrieved_metadata.iloc[0]['Life stage'],
                'microscope': retrieved_metadata.iloc[0]['Microscope'],
                'neuron': retrieved_metadata.iloc[0]['Neuron'],
                'notes': retrieved_metadata.iloc[0]['Notes'],
                'num_eggs': retrieved_metadata.iloc[0][
                        'Number of eggs visible'],
                'microscope_objective': retrieved_metadata.iloc[0][
                        'Objective'],
                'purpose': retrieved_metadata.iloc[0]['Purpose of Experiment'],
                'worm_strain': retrieved_metadata.iloc[0]['Worm Strain'],
                'head_orientation': retrieved_metadata.iloc[0][
                        'Worm head orientation'],
                'vulva_orientation': retrieved_metadata.iloc[0][
                        'Worm vulva orientation']}

        return metadata_dict


class PrettySafeLoader(yaml.SafeLoader):  # not sure if I need this anymore
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


if __name__ == '__main__':
    test_trial = StrainPropagationTrial(
            '/Users/adam/Documents/''SenseOfTouchResearch/'
            'SSN_data/20181220/SSN_126_001.nd2',
            load_images=True, overwrite_metadata=False)
    my_metadata = test_trial.metadata
