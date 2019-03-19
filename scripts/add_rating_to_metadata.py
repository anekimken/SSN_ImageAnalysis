#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 15:53:13 2019

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

metadata_files = glob.iglob(wdir + '**/metadata.yaml')
for file in metadata_files:
    print(file)
    with open(file, 'r') as yamlfile:
        metadata = yaml.load(yamlfile)
    if ('trial_rating' not in metadata) or (
            metadata['trial_rating'] == 'No rating given'):

        experiment_id = file[-25:-14]

        if int(experiment_id[4:7]) > 1: # can change this to skip older trials
            current_id_cell = metadata_worksheet.find(experiment_id)
            row_of_metadata = metadata_worksheet.row_values(
                    current_id_cell.row)
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
