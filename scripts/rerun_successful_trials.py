#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script that puts all trials with status "Strain calculated" into the queue and
then runs the queue so they can be analyzed again with the latest code. The
main idea is to make sure bug fixes make their way to all the trials, not just
trials after the bug got fixed.

Created on Thu Apr  4 10:36:26 2019

@author: adam
"""

import yaml
import glob
import os
import pandas as pd
import warnings
import pathlib
import scripts.run_queue as run_queue
from strain_propagation_trial import StrainPropagationTrial


def add_successful_trials_to_queue():
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))
    yaml.add_constructor(u'tag:yaml.org,2002:python/tuple',
                         construct_python_tuple, Loader=yaml.SafeLoader)

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

    # Load up the queue
    the_queue = file_paths['analysis_dir'] + 'analysis_queue.yaml'
    queue_file_size = os.stat(the_queue).st_size
    if queue_file_size > 0:
        warnings.warn('Queue not empty, might result in running the same '
                      'trial more than once or overwriting new results.')
    for index, trial in trials_with_strain_calc.iterrows():

        batch_param_history_file = (file_paths['analysis_dir'] +
                                    'AnalyzedData/' +
                                    trial['Experiment_id'] +
                                    '/trackpyBatchParamsHistory.yaml')

        with open(batch_param_history_file, 'r') as param_file:
            entire_history = yaml.safe_load_all(param_file)
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

    # Actually do the particle finding for these trials
    run_queue.run_queue()

    # Do the strain calculation for these trials
    for index, this_trial in trials_with_strain_calc.iterrows():
        trial = StrainPropagationTrial()
        trial.analyzed_data_location = pathlib.Path(
                file_paths['analysis_dir'] + 'AnalyzedData/' +
                this_trial['Experiment_id'] + '/')
        trial.batch_data_file = trial.analyzed_data_location.joinpath(
                 'trackpyBatchResults.yaml')
        with open(trial.batch_data_file, 'r') as yamlfile:
            linked_mitos_dict = yaml.safe_load(yamlfile)
            trial.linked_mitos = pd.DataFrame.from_dict(
                    linked_mitos_dict, orient='index')
        trial.metadata = this_trial.to_dict()
        trial.calculate_strain()


if __name__ == '__main__':
    # execute as script
    add_successful_trials_to_queue()
