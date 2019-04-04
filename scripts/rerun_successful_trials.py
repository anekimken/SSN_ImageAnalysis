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
import ssn_image_analysis_gui_controller as ssn_cont

controller = ssn_cont.StrainGUIController(headless=True)

# TODO: move file path to config file
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
