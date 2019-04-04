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

with open('config.yaml', 'r') as config_file:
    file_paths = yaml.safe_load(config_file)
queue_location = file_paths['analysis_dir']
the_queue = queue_location + 'analysis_queue.yaml'
controller = ssn_cont.StrainGUIController(headless=True)

with open(the_queue, 'r') as queue_file:
            entire_queue = yaml.load_all(queue_file)
            queue_length = len(list(entire_queue))

print('Running queue with', queue_length, 'items.')
error_list = []
queue_start_time = time.time()
while True:  # queue_length > 0:
    queue_file_size = os.stat(the_queue).st_size
    if queue_file_size > 0:
        try:
            controller.run_queue_item(queue_location)
        except AttributeError as error:
            if error.args[0] == "'DataFrame' object has no attribute 'dtype'":
                print('dataframe error')
                error_list.append((controller.trial.experiment_id, error))
                # Remove first trial in queue, since we're done with it
                with open(the_queue, 'r') as queue_file:
                    old_queue = yaml.load_all(queue_file)
                    new_queue = [
                            item for item in old_queue
                            if item['experiment_id'] !=
                            controller.trial.experiment_id]

                with open(the_queue, 'w') as queue_file:
                    yaml.dump_all(new_queue, queue_file, explicit_start=True)
                pass
        # update queue length variable
        with open(the_queue, 'r') as queue_file:
            entire_queue = yaml.load_all(queue_file)
            queue_length = len(list(entire_queue))
    else:
        print('Finished queue in',
              str(round(time.time() - queue_start_time)), 'seconds.')
        print("Errors occurred on", error_list)
        break
