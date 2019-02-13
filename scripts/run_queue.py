#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:42:09 2019

@author: adam
"""

import yaml
import os
import time
import ssn_image_analysis_gui_controller as ssn_cont

queue_location = ('/Users/adam/Documents/SenseOfTouchResearch/'
                  'SSN_ImageAnalysis/analysis_queue.yaml')
controller = ssn_cont.StrainGUIController(headless=True)

with open(queue_location, 'r') as queue_file:
            entire_queue = yaml.load_all(queue_file)
            queue_length = len(list(entire_queue))

print('Running queue with', queue_length, 'items.')
while True:  # queue_length > 0:
    queue_file_size = os.stat(queue_location).st_size
    if queue_file_size > 0:
        controller.run_queue_item(queue_location)

        # update queue length variable
        with open(queue_location, 'r') as queue_file:
            entire_queue = yaml.load_all(queue_file)
            queue_length = len(list(entire_queue))
    else:
        print('Queue is empty.')
    time.sleep(10)
