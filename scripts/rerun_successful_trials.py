#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script that puts all trials with status "Strain calculated" into the queue
so they can be analyzed again with the latest code. The main idea is to make
sure bug fixes make their way to all the trials, not just trials after the
bug got fixed.

Created on Thu Apr  4 10:36:26 2019

@author: adam
"""

import yaml
import glob
import pandas as pd
import run_queue


def add_successful_trials_to_queue():
#    controller = ssn_cont.StrainGUIController(headless=True)

    with open('config.yaml', 'r') as config_file:
        file_paths = yaml.safe_load(config_file)
    analyzed_data_dir = file_paths['analysis_dir'] + 'AnalyzedData/'

    # get all the metadata
    experiment_days = glob.iglob(analyzed_data_dir + '*')
    all_metadata = []
    for day in experiment_days:
        try:
            with open(day + '/metadata.yaml') as metadata_file:
                all_metadata.append(yaml.safe_load(metadata_file))
        except FileNotFoundError:
            pass
    metadata_df = pd.DataFrame(all_metadata)

    # filter out trials we don't want
    trials_with_strain_calc = metadata_df.loc[
            metadata_df['analysis_status'] == 'Strain calculated']

    the_queue = file_paths['analysis_dir'] + 'analysis_queue.yaml'
    for index, trial in trials_with_strain_calc.iterrows():
        print(trial['Experiment_id'])

        batch_param_history_file = (file_paths['analysis_dir'] +
                                    'AnalyzedData/' +
                                    trial['Experiment_id'] +
                                    '/trackpyBatchParamsHistory.yaml')

        with open(batch_param_history_file, 'r') as param_file:
            entire_history = yaml.load_all(param_file)
            last_params = None
            for last_params in entire_history:
                pass  # pass until we get to most recent

        param_dict = {'experiment_id': trial['Experiment_id'],
                      'roi': last_params['roi'],
                      'gaussian_width': last_params['noise_size'],
                      'particle_z_diameter': last_params['diameter'][0],
                      'particle_xy_diameter': last_params['diameter'][1],
                      'brightness_percentile': last_params['percentile'],
                      'min_particle_mass': last_params['minmass'],
                      'bottom_slice': last_params['bottom_slice'],
                      'top_slice': last_params['top_slice'],
                      'tracking_seach_radius': last_params[
                              'tracking_seach_radius'],
                      'last_timepoint': last_params['last_timepoint']}
        if 'notes' in last_params:
            param_dict['notes'] = last_params['notes']

        with open(the_queue, 'a') as output_file:
                yaml.dump(param_dict, output_file, explicit_start=True)


if __name__ == '__main__':
    # execute as script
    add_successful_trials_to_queue()
    run_queue.run_queue()
