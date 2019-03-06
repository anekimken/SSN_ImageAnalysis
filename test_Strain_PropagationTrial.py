#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 17:01:49 2019

@author: Adam
"""

# TODO: DOCSTRINGS!!!!
import unittest
import datetime

from strain_propagation_trial import StrainPropagationTrial


class TestLoadTrial(unittest.TestCase):
    def setUp(self):
        self.filename = ('/Users/adam/Documents/SenseOfTouchResearch/'
                         'SSN_data/20181220/SSN_126_001.nd2')
        self.test_cases = [
                (False, False),
                (False, True),
                (True, False),
                (True, True)]

    def test_init(self):
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

    # TODO: make a test trial that can run really fast through everything


"""
Things to test:
    * Asterisk indicates highest priorities

    Using actual data:
        load_trial using actual nd2 data and actual metadata

    Using artificial data:
        find_mitos_one_stack
        *find_mitos_current_trial
        link_existing_particles
        *calculate_strain

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
