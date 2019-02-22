#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:02:05 2019

@author: adam
"""


import yaml
strain_files = glob.iglob(wdir + '**/strain.yaml')

for file in metadata_files:
    print(file)
    with open(file, 'r') as yamlfile:
        metadata = yaml.load(yamlfile)
    if 'trial_rating' not in metadata:

        experiment_id = file[-25:-14]

        current_id_cell = metadata_worksheet.find(experiment_id)
        row_of_metadata = metadata_worksheet.row_values(current_id_cell.row)
        row_of_keys = metadata_worksheet.row_values(1)
        gdrive_metadata_dict = dict(zip(row_of_keys, row_of_metadata))
#        print(gdrive_metadata_dict)

        try:
            metadata['trial_rating'] = gdrive_metadata_dict['Trial rating']
            print(metadata['trial_rating'])
            with open(file, 'w') as output_file:
                yaml.dump(metadata, output_file,  # create file
                          explicit_start=True, default_flow_style=False)
        except KeyError:
            print("didn't find a rating value for ", experiment_id)
