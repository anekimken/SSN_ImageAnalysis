#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 10:45:42 2019

@author: adam
"""


import numpy as np
import tkinter as tk
import threading
from multiprocessing import Process, Queue
import glob
import yaml

import strain_propagation_trial as ssn_trial
import strain_propagation_view as ssn_view
import ssn_image_analysis_gui_controller


trial = ssn_trial.StrainPropagationTrial()
print('Loading file')
trial.load_trial('/Users/adam/Documents/SenseOfTouchResearch/SSN_data/'
                 '20181220/SSN_126_001.nd2',
                 load_images=True, overwrite_metadata=False)
print('running batch')
trial.run_batch(gaussian_width=3,
                particle_z_diameter=21,
                particle_xy_diameter=15,
                brightness_percentile=50,
                min_particle_mass=200,
                bottom_slice=1,
                top_slice=69,
                tracking_seach_radius=40,
                last_timepoint=3)
trajectories_found = trial.linked_mitos
dict_for_saving = trajectories_found.reset_index(drop=True).to_dict(orient='index')
