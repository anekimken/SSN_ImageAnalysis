#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:13:42 2019

@author: adam

Goes through google drive metadata sheet and adds the triak_rating field to the
metadata.yaml file for use by user during analysis
"""

import yaml
import glob
import gspread
from oauth2client.service_account import ServiceAccountCredentials


wdir = ('/Users/adam/Documents/SenseOfTouchResearch/'
        'SSN_ImageAnalysis/AnalyzedData/')

# Access google drive spreadsheet
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
keyfile = ('/Users/adam/Documents/SenseOfTouchResearch/'
           'SSN_ImageAnalysis/SenseOfTouchResearch-e5927f56c4d0.json')
credentials = ServiceAccountCredentials.from_json_keyfile_name(
        keyfile, scope)
gspread_client = gspread.authorize(credentials)
metadata_spread = gspread_client.open_by_key(
        '1LsTdPBOW79XSkk5DJv2ckiVTvpofAOOd_dJL4cgtxBQ')
metadata_worksheet = metadata_spread.worksheet("MetadataResponses")

metadata_files = glob.glob(wdir + '**/metadata.yaml')

for file in reversed(metadata_files):
    print(file)
    with open(file, 'r') as yamlfile:
        metadata = yaml.load(yamlfile)
    if 'pressure_kPa' not in metadata:

        experiment_id = file[-25:-14]

        current_id_cell = metadata_worksheet.find(experiment_id)
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
        if len(pressure_kPa) > 0:
            gdrive_metadata_dict['pressure_kPa'] = pressure_kPa

        try:
            assert (metadata['num_timepoints'] == len(pressure_kPa) or
                    metadata['num_timepoints'] == len(pressure_kPa) + 1)
            metadata['pressure_kPa'] = gdrive_metadata_dict['pressure_kPa']
            print(metadata['pressure_kPa'])
            with open(file, 'w') as output_file:
                yaml.dump(metadata, output_file,  # create file
                          explicit_start=True, default_flow_style=False)
        except KeyError:
            print("didn't find pressure values for ", experiment_id)
    else:
        print('Already have pressure values')
