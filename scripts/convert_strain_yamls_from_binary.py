#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:02:05 2019

@author: adam
"""

import glob
import yaml
import pandas as pd

wdir = ('/Users/adam/Documents/SenseOfTouchResearch/'
        'SSN_ImageAnalysis/AnalyzedData/')
strain_files = glob.iglob(wdir + '**/old_strain.yaml')

for file in strain_files:
    print(file)
    result_folder = file[:-15]
    with open(file, 'r') as yamlfile:
        strain = yaml.load(yamlfile)

    mito_data_file = result_folder + 'trackpyBatchResults.yaml'
    with open(mito_data_file, 'r') as yamlfile:
        linked_mitos_dict = yaml.load(yamlfile)
        mitos_df = pd.DataFrame.from_dict(
                    linked_mitos_dict, orient='index')
#    print(len(linked_mitos_dict))
    num_frames = mitos_df['frame'].nunique()
    ycoords = []
    for stack in range(num_frames):
        # sort mitochondria in this stack by y values
        current_stack = mitos_df.loc[mitos_df['frame'] == stack]
        sorted_stack = current_stack.sort_values(['y'])
        sorted_stack.reset_index(inplace=True, drop=True)
        ycoords.append(list(sorted_stack['y']))
    strain_list = strain.tolist()

    if len(ycoords) == len(strain_list):
        results_dict = {'strain': strain_list, 'ycoords': ycoords}
        backup_file = result_folder + 'old_strain.yaml'
        with open(backup_file, 'w') as yaml_file:
            yaml.dump(strain, yaml_file,
                      explicit_start=True, default_flow_style=False)

        fresh_file = result_folder + 'strain.yaml'
        with open(fresh_file, 'w') as yaml_file:
            yaml.dump(results_dict, yaml_file,
                      explicit_start=True, default_flow_style=False)
    else:
        print('ycoords and strain have different number of frames.'
              ' try recalculating strain.')
