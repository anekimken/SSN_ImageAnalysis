#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 21:59:37 2019

@author: adam
"""

import glob
import yaml

base_dir = '/Users/adam/Documents/SenseOfTouchResearch/'
metadata_location = (base_dir + 'SSN_ImageAnalysis/AnalyzedData/')

trackpy_param_test_history = glob.glob(metadata_location +
                                       '/*/trackpyParamTestHistory.yaml')
print(trackpy_param_test_history)

for file in trackpy_param_test_history:
    with open(file, 'r') as yamlfile:
        entire_history = yaml.load_all(yamlfile)
        trackpy_locate_params = None
        for trackpy_locate_params in entire_history:  # get latest params
            pass
    print(trackpy_locate_params)
