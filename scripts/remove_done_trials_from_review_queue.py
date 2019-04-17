#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:10:35 2019

@author: adam

Goes through the review queue and removes all trials where strain has already
been calculated.
"""

import yaml
import pathlib


def construct_python_tuple(self, node):
    return tuple(self.construct_sequence(node))


yaml.add_constructor(u'tag:yaml.org,2002:python/tuple',
                     construct_python_tuple, Loader=yaml.SafeLoader)

with open('config.yaml', 'r') as config_file:
    file_paths = yaml.safe_load(config_file)
analyzed_data_dir = pathlib.Path(file_paths['analysis_dir'])
review_queue = analyzed_data_dir / 'review_queue.yaml'
type(review_queue)

with open(review_queue, 'r') as review_queue:
    queue = yaml.safe_load_all(review_queue)
    trials_in_review_queue = []
    for trial in queue:
        trials_in_review_queue.append(trial['experiment_id'])

review_queue = analyzed_data_dir / 'review_queue.yaml'
for trial in trials_in_review_queue:
    metadata_file_path = (analyzed_data_dir / 'AnalyzedData' /
                          trial / 'metadata.yaml')
    with open(metadata_file_path, 'r') as metadata_file:
        metadata = yaml.safe_load(metadata_file)

    if metadata['analysis_status'] == 'Strain calculated':
        print(type(review_queue))
        with open(review_queue, 'r') as queue_file:
            old_queue = yaml.safe_load_all(queue_file)
            new_q = [item for item in old_queue
                     if item['experiment_id'] != trial]

        with open(review_queue, 'w') as queue_file:
            yaml.dump_all(new_q, queue_file, explicit_start=True)
