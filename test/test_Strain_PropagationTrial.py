#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 17:01:49 2019

@author: Adam
"""

# TODO: DOCSTRINGS!!!!
import unittest
import datetime
import numpy as np
import pandas as pd
import pathlib

from strain_propagation_trial import StrainPropagationTrial
import scripts.create_test_dataset as create_test_dataset


@unittest.skip("skipping load_trial tests")
class TestLoadTrial(unittest.TestCase):
    def setUp(self):
        self.filename = ('/Users/adam/Documents/SenseOfTouchResearch/'
                         'SSN_data/20181220/SSN_126_001.nd2')
        self.test_cases = [(False, False)]
#                           (False, True)]

    def test_load_trial(self):
        for load_image_flag, overwrite_metadata_flag in self.test_cases:
            print("Checking case :", load_image_flag,
                  overwrite_metadata_flag)
            trial = StrainPropagationTrial()
            trial.load_trial(
                    self.filename,
                    load_images=load_image_flag,
                    overwrite_metadata=overwrite_metadata_flag)

            self.assertIsInstance(trial.metadata, dict)
            self.assertEqual(trial.metadata["Experiment_id"],
                             "SSN_126_001")
            bleach_datetime = datetime.datetime(2018, 12, 17, 9, 30, 0)
            self.assertEqual(trial.metadata["bleach_time"],
                             bleach_datetime)
            self.assertEqual(trial.metadata["cultivation_temp"], 20)
            self.assertEqual(trial.metadata["device_ID"], 'Y4P7-7')
            self.assertEqual(trial.metadata["head_orientation"],
                             'Headfirst')
            self.assertEqual(trial.metadata["microscope"],
                             'Spinning Disk Confocal in CSIF Shriram')
            self.assertEqual(trial.metadata["microscope_channel"],
                             'SDC 561-ALN')
            self.assertEqual(trial.metadata["microscope_objective"],
                             '40x oil')
            self.assertEqual(trial.metadata["neuron"], 'PVM')
            self.assertEqual(trial.metadata["notes"],
                             'Neuron not all in one plane. Bleached after '
                             'a few trials, especially mitochondria that '
                             'were deeper into the worm. 11 total stacks, '
                             '4 with actuation: 0, 0, 300, 0, 300, 0, 300,'
                             ' 0, 300, 0, 0 kPa.')
            self.assertEqual(trial.metadata["num_eggs"], 3)
            self.assertEqual(trial.metadata["pixel_microns"], 0.275)
            self.assertEqual(trial.metadata["purpose"],
                             'Baseline measurement')
            self.assertEqual(trial.metadata["slice_height_pix"], 1200)
            self.assertEqual(trial.metadata["slice_width_pix"], 600)
            timestamp_datetime = datetime.datetime(2018, 12, 20, 11, 5, 37)
            self.assertEqual(trial.metadata["timestamp"],
                             timestamp_datetime)
            self.assertEqual(trial.metadata["total_images"], 759)
            self.assertEqual(trial.metadata["user"],
                             'anekimke@stanford.edu')
            self.assertEqual(trial.metadata["vulva_orientation"], 'West')
            self.assertEqual(trial.metadata["worm_life_stage"], 'YA')
            self.assertEqual(trial.metadata["worm_strain"], 'NM3573')


# @unittest.skip("skipping run_batch tests")
class TestFindMitosArtificialData(unittest.TestCase):
    def setUp(self):
        self.mito_coords = pd.DataFrame([[30, 200, 350, 0],
                                         [30, 250, 350, 0],
                                         [30, 300, 350, 0],
                                         [30, 350, 350, 0],
                                         [30, 200, 350, 1],
                                         [30, 250, 360, 1],
                                         [30, 300, 360, 1],
                                         [30, 350, 350, 1],
                                         [30, 200, 350, 2],
                                         [30, 250, 350, 2],
                                         [30, 300, 350, 2],
                                         [30, 350, 350, 2]],
                                        columns=['z', 'y', 'x', 'frame'])
        self.stack_size = (3, 61, 1200, 600)
        self.images = np.ndarray(self.stack_size)
        self.images = create_test_dataset.create_test_stack(
                    self.mito_coords, size=self.stack_size)
        self.trial = StrainPropagationTrial()
        self.trial.metadata = {'pressure_kPa': [0, 300, 0]}
        self.trial.analyzed_data_location = pathlib.Path('./test/results_data')
        data_loc = self.trial.analyzed_data_location
        self.trial.batch_history_file = data_loc.joinpath(
                'trackpyBatchParamsHistory.yaml')
        self.test_cases = [{'images_ndarray': self.images,
                            'roi': [300, 100, 400, 450],
                            'gaussian_width': 3,
                            'particle_z_diameter': 21,
                            'particle_xy_diameter': 15,
                            'brightness_percentile': 50,
                            'min_particle_mass': 50,
                            'bottom_slice': 0,
                            'top_slice': 61,
                            'tracking_seach_radius': 20,
                            'last_timepoint': 3,
                            'notes': 'Artificial data test'},
                           {'images_ndarray': self.images,
                            'roi': [200, 100, 500, 450],
                            'gaussian_width': 3,
                            'particle_z_diameter': 21,
                            'particle_xy_diameter': 15,
                            'brightness_percentile': 50,
                            'min_particle_mass': 50,
                            'bottom_slice': 0,
                            'top_slice': 61,
                            'tracking_seach_radius': 20,
                            'last_timepoint': 3,
                            'notes': 'Artificial data test'}]

    def test_batch_find_entire_trial(self):
        for case in self.test_cases:
            self.trial.run_batch(**case)
            linked = self.trial.linked_mitos.reset_index(drop=True)
            linked.sort_values(by=['frame', 'particle'], inplace=True)
            linked.reset_index(drop=True, inplace=True)
            pd.testing.assert_frame_equal(linked[['x', 'y', 'z', 'frame']],
                                          self.mito_coords,
                                          check_less_precise=True,
                                          check_like=True,
                                          check_dtype=False)


class TestCalcStrainArtificialData(unittest.TestCase):
    def setUp(self):
        self.trial = StrainPropagationTrial()
        self.trial.analyzed_data_location = pathlib.Path('./test/results_data')

        self.test_cases = [
                [pd.DataFrame([[30, 200, 350, 0, 0],
                               [30, 250, 350, 0, 1],
                               [30, 300, 350, 0, 2],
                               [30, 350, 350, 0, 3],
                               [30, 200, 350, 1, 0],
                               [30, 250, 350, 1, 1],
                               [30, 300, 350, 1, 2],
                               [30, 350, 350, 1, 3],
                               [30, 200, 350, 2, 0],
                               [30, 250, 350, 2, 1],
                               [30, 300, 350, 2, 2],
                               [30, 350, 350, 2, 3]],
                              columns=['z', 'y', 'x', 'frame', 'particle']),
                 {'pressure_kPa': [0, 300, 0]}, [[0, 0, 0],
                                                 [0, 0, 0],
                                                 [0, 0, 0]]],

                [pd.DataFrame([[30, 200, 350, 0, 0],
                               [30, 250, 350, 0, 1],
                               [30, 300, 350, 0, 2],
                               [30, 350, 350, 0, 3],
                               [30, 200, 350, 1, 0],
                               [30, 250, 360, 1, 1],
                               [30, 300, 360, 1, 2],
                               [30, 350, 350, 1, 3],
                               [30, 200, 350, 2, 0],
                               [30, 250, 350, 2, 1],
                               [30, 300, 350, 2, 2],
                               [30, 350, 350, 2, 3]],
                              columns=['z', 'y', 'x', 'frame', 'particle']),
                 {'pressure_kPa': [0, 300, 0]}, [[0, 0, 0],
                                                 [0.0198039, 0, 0.0198039],
                                                 [0, 0, 0]]],

                [pd.DataFrame([[30, 200, 350, 0, 0],
                               [30, 250, 350, 0, 1],
                               [30, 300, 350, 0, 2],
                               [30, 350, 350, 0, 3],
                               [30, 200, 350, 1, 0],
                               [30, 260, 350, 1, 1],
                               [30, 300, 350, 1, 2],
                               [30, 350, 350, 1, 3],
                               [30, 200, 350, 2, 0],
                               [30, 250, 350, 2, 1],
                               [30, 300, 350, 2, 2],
                               [30, 350, 350, 2, 3]],
                              columns=['z', 'y', 'x', 'frame', 'particle']),
                 {'pressure_kPa': [0, 300, 0]}, [[0, 0, 0],
                                                 [0.2, -0.2, 0],
                                                 [0, 0, 0]]],
                [pd.DataFrame([[30, 200, 350, 0, 0],
                               [30, 250, 350, 0, 1],
                               [30, 300, 350, 0, 2],
                               [30, 350, 350, 0, 3],
                               [30, 200, 350, 1, 0],
                               [20, 250, 350, 1, 1],
                               [20, 300, 350, 1, 2],
                               [30, 350, 350, 1, 3],
                               [30, 200, 350, 2, 0],
                               [30, 250, 350, 2, 1],
                               [30, 300, 350, 2, 2],
                               [30, 350, 350, 2, 3]],
                              columns=['z', 'y', 'x', 'frame', 'particle']),
                 {'pressure_kPa': [0, 300, 0]}, [[0, 0, 0],
                                                 [0.0198039, 0, 0.0198039],
                                                 [0, 0, 0]]]]
#                [pd.DataFrame([[30, 200, 350, 0, 0],
#                               [27, 204, 350, 0, 1],
#                               [30, 200, 350, 1, 0],
#                               [33, 204, 350, 1, 1],
#                               [30, 200, 350, 2, 0],
#                               [33, 204, 350, 2, 1]],
#                              columns=['z', 'y', 'x', 'frame', 'particle']),
#                 {'pressure_kPa': [0, 300, 0]}, [[0.25],
#                                                 [-.25],
#                                                 [0.25]]]] # turns out this is just rotation, not deformation
#            [pd.DataFrame([[30, 200, 350, 0, 0],
#                               [30, 250, 350, 0, 1],
#                               [30, 300, 350, 0, 2],
#                               [30, 350, 350, 0, 3],
#                               [30, 200, 350, 1, 0],
#                               [20, 250, 350, 1, 1],
#                               [20, 300, 350, 1, 2],
#                               [30, 350, 350, 1, 3],
#                               [30, 200, 350, 2, 0],
#                               [20, 250, 350, 2, 1],
#                               [20, 300, 350, 2, 2],
#                               [30, 350, 350, 2, 3]],
#                              columns=['z', 'y', 'x', 'frame', 'particle']),
#                 {'pressure_kPa': [0, 300, 0]}, [[0.1, 0, 0.1],
#                                                 [0.1, 0, 0.1],
#                                                 [0.1, 0, 0.1]]]

    def test_strain_calc_with_artificial_data(self):
        for linked_mitos, pressure, desired_output in self.test_cases:
            self.trial.linked_mitos = linked_mitos
            self.trial.metadata = pressure
            self.trial.calculate_strain()
            np.testing.assert_almost_equal(self.trial.strain, desired_output)
            print('tested a case')


"""
Things to test:
    * Asterisk indicates highest priorities

    Using actual data:
        load_trial using actual nd2 data and actual metadata

    Using artificial data:
        find_mitos_one_stack
        #*find_mitos_current_trial
        link_existing_particles
        *calculate_strain -> bug

    Using a two sets of artificial data:
        run_multiple_files this might take too long
        add_trial_to_queue
        run_queue/run_queue_item
        remove_from_review_queue

    Results files from artificial data:
        metadata.yaml
        trackpyBatchParams.yaml
        *trackpyBatchResults.yaml
        unlinkedTrackpyBatchResults.yaml
        *strain.yaml

"""

if __name__ == '__main__':
    unittest.main()
