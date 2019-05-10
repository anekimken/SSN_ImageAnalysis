#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:32:39 2019

@author: Adam Nekimken
"""

import time
import numpy as np
import tkinter as tk
import pandas as pd
from scipy import spatial
import matplotlib as mpl
import glob
import yaml
import pims
import matplotlib.pyplot as plt
import fast_histogram
import pathlib
from pandastable import TableModel

import strain_propagation_trial as ssn_trial
import strain_propagation_view as ssn_view


class StrainGUIController:
    """Controller for running strain propagation analysis program.

    Uses Model-View-Controller architecture for doing analysis of data for
    the strain propagation project.
    """
    def __init__(self, headless=False):
        self.root = tk.Tk()
        with open('config.yaml', 'r') as config_file:
            self.file_paths = yaml.safe_load(config_file)

        def construct_python_tuple(self, node):
            return tuple(self.construct_sequence(node))

        yaml.add_constructor(u'tag:yaml.org,2002:python/tuple',
                             construct_python_tuple, Loader=yaml.SafeLoader)

        # Instantiate Model
        self.trial = ssn_trial.StrainPropagationTrial()

        if headless is not True:
            # Instantiate View
            self.gui = ssn_view.SSN_analysis_GUI(self.root)

            # Bind UI elements to functions
            # Bind keys to root
            self.root.bind("<Down>", self.on_arrow_key)
            self.root.bind("<Up>", self.on_arrow_key)

            # Load trial tab
            self.gui.file_load_frame.load_trial_button.bind(
                    "<ButtonRelease-1>", func=self.load_trial)
            self.gui.file_load_frame.run_multiple_files_button.bind(
                    "<ButtonRelease-1>", func=self.run_multiple_files)
            self.gui.file_load_frame.file_tree.bind(
                    '<<TreeviewSelect>>', func=self.on_file_selection_changed)

            # Analysis frame
            self.gui.analyze_trial_frame.plot_canvas._tkcanvas.bind(
                    "<ButtonPress-1>", self.on_click_image)
            self.gui.analyze_trial_frame.plot_canvas._tkcanvas.bind(
                    "<B1-Motion>", self.on_drag_roi)
            self.gui.analyze_trial_frame.plot_canvas._tkcanvas.bind(
                    "<ButtonRelease-1>", self.on_click_image_release)
            self.gui.analyze_trial_frame.plot_canvas._tkcanvas.bind(
                    "<Enter>", self._bound_to_mousewheel)
            self.gui.analyze_trial_frame.plot_canvas._tkcanvas.bind(
                    "<Leave>", self._unbound_to_mousewheel)
            self.gui.analyze_trial_frame.side_canvas._tkcanvas.bind(
                    "<Enter>", self._bound_to_mousewheel)
            self.gui.analyze_trial_frame.side_canvas._tkcanvas.bind(
                    "<Leave>", self._unbound_to_mousewheel)
            self.gui.analyze_trial_frame.clear_roi_btn.bind(
                    "<ButtonRelease-1>", self.clear_roi)
            self.gui.analyze_trial_frame.update_image_btn.bind(
                    "<ButtonRelease-1>", func=self.update_inspection_image)
            self.gui.analyze_trial_frame.slice_selector.bind(
                    "<ButtonRelease-1>",
                    func=self.spinbox_delay_then_update_image)
            self.gui.analyze_trial_frame.timepoint_selector.bind(
                    "<ButtonRelease-1>",
                    func=self.spinbox_delay_then_update_image)
#            self.gui.analyze_trial_frame.min_pixel_disp.bind(
#                    "<ButtonRelease-1>",
#                    func=self.spinbox_delay_then_update_image)
#            self.gui.analyze_trial_frame.max_pixel_disp.bind(
#                    "<ButtonRelease-1>",
#                    func=self.spinbox_delay_then_update_image)
            self.gui.analyze_trial_frame.unbind_all("<Return>")
            self.gui.analyze_trial_frame.min_pixel_disp.bind(
                    "<Return>",
                    func=self.spinbox_delay_then_update_image)
            self.gui.analyze_trial_frame.max_pixel_disp.bind(
                    "<Return>",
                    func=self.spinbox_delay_then_update_image)
            self.gui.analyze_trial_frame.test_param_button.bind(
                    "<ButtonRelease-1>", func=self.find_mitos_one_stack)
            self.gui.analyze_trial_frame.full_analysis_button.bind(
                    "<ButtonRelease-1>", func=self.find_mitos_current_trial)
            self.gui.analyze_trial_frame.status_dropdown[
                    'values'] = self.trial.STATUSES
            self.gui.analyze_trial_frame.status_dropdown.bind(
                    "<<ComboboxSelected>>", func=self.update_status)
            self.gui.analyze_trial_frame.plot_labels_drop.bind(
                    "<<ComboboxSelected>>", func=self.update_inspection_image)
            self.gui.analyze_trial_frame.link_mitos_button.bind(
                    "<ButtonRelease-1>", func=self.link_existing_particles)
            self.gui.analyze_trial_frame.add_to_queue_btn.bind(
                    "<ButtonRelease-1>", func=self.add_trial_to_queue)
            self.gui.analyze_trial_frame.remove_q_result.bind(
                    "<ButtonRelease-1>", func=self.remove_from_review_queue)
            self.gui.analyze_trial_frame.calc_strain_button.bind(
                    "<ButtonRelease-1>", func=self.calculate_strain)
            self.gui.analyze_trial_frame.analysis_notebook.bind(
                    "<<NotebookTabChanged>>", func=self.update_histogram)

            # Queue frame
            self.gui.queue_frame.run_queue_button.bind(
                    "<ButtonRelease-1>", func=self.run_queue)

            # Plotting frame
            self.gui.plot_results_frame.progress_plot_button.bind(
                    "<ButtonRelease-1>", func=self.get_analysis_progress)
            self.gui.plot_results_frame.plot_strain_one_trial_button.bind(
                    "<ButtonRelease-1>", func=self.plot_existing_strain)
            self.gui.plot_results_frame.plot_strain_by_actuation_btn.bind(
                    "<ButtonRelease-1>", func=self.plot_existing_strain)
            self.gui.plot_results_frame.plot_xz_disp_btn.bind(
                    "<ButtonRelease-1>", func=self.plot_xz_displacements)

    def run(self):
        self.root.title("SSN Image Analysis")
        self.root.mainloop()

    def load_trial(self, event=None):
        """Loads the data for this trial from disk and sends us to the
        inspection tab to start the analysis
        """
        # MAYBE: add popup window saying file is loading
        start_time = time.time()
        self.trial = ssn_trial.StrainPropagationTrial()
        print('Loading file', self.file_list[0][0])

        fl_fr = self.gui.file_load_frame
        load_images = fl_fr.load_images_box.instate(['selected'])
        overwrite_metadata = fl_fr.overwrite_metadata_box.instate(['selected'])

        self.trial.load_trial(self.file_list[0][0],
                              load_images=load_images,
                              overwrite_metadata=overwrite_metadata)
        if 'analysis_status' not in self.trial.metadata:
            self.trial.metadata['analysis_status'] = 'Not started'
            self.trial.write_metadata_to_yaml(self.trial.metadata)
        current_status = self.trial.metadata['analysis_status']
        self.gui.analyze_trial_frame.status_dropdown.set(current_status)
        if load_images is True or overwrite_metadata is True:
            min_pixel = int(np.amin(self.trial.image_array))
            min_spinbox = self.gui.analyze_trial_frame.min_pixel_disp
            min_spinbox.delete(0, 'end')
            min_spinbox.insert(0, str(min_pixel))
            max_pixel = int(np.amax(self.trial.image_array))
            max_spinbox = self.gui.analyze_trial_frame.max_pixel_disp
            max_spinbox.delete(0, 'end')
            max_spinbox.insert(0, str(max_pixel))
            if self.trial.linked_mitos is not None:
                if self.trial.linked_mitos.empty is False:
                    self.gui.analyze_trial_frame.plot_labels_drop.set(
                        'Plot trajectories')

        # load inspection image on inspection tab and related parameters
        self.roi = self.trial.latest_test_params['roi']
        self.update_inspection_image()

        image_stack_size = self.trial.metadata['stack_height']
        num_timepoints = self.trial.metadata['num_timepoints']
        self.gui.analyze_trial_frame.slice_selector.config(
                values=list(range(1, image_stack_size + 1)))
        self.gui.analyze_trial_frame.btm_slice_selector.config(
                values=list(range(1, image_stack_size + 1)))
        self.gui.analyze_trial_frame.top_slice_selector.config(
                values=list(range(1, image_stack_size + 1)))
        self.gui.analyze_trial_frame.last_time_selector.config(
                values=list(range(1, num_timepoints + 1)))
        self.gui.analyze_trial_frame.timepoint_selector.config(
                values=list(range(1, num_timepoints + 1)))
        self.gui.analyze_trial_frame.metadata_notes.config(
                text=self.trial.metadata['notes'])

        self.gui.analyze_trial_frame.stack_height_label.config(
                text=('Stack height: ' + str(image_stack_size)))
        self.gui.analyze_trial_frame.neuron_id_label.config(
                text=('Neuron: ' + self.trial.metadata['neuron']))
        self.gui.analyze_trial_frame.vulva_side_label.config(
                text=('Vulva side: ' +
                      self.trial.metadata['vulva_orientation']))
        self.gui.analyze_trial_frame.notes_entry.delete(0, tk.END)
        if 'notes' in self.trial.latest_test_params:
            self.gui.analyze_trial_frame.notes_entry.insert(
                    tk.END, self.trial.latest_test_params['notes'])

        self._display_last_test_params()

        # load param history into text field
        if self.trial.batch_history_file.is_file():
            with open(self.trial.batch_history_file, 'r') as hist_yaml:
                param_history = hist_yaml.read()
                hist_text = self.gui.analyze_trial_frame.param_history
                hist_text.config(state=tk.NORMAL)
                hist_text.delete(1.0, tk.END)
                hist_text.insert(tk.END, param_history)
                hist_text.config(state=tk.DISABLED)
                hist_text.yview_moveto(1)

        if self.trial.metadata['Experiment_id'] in fl_fr.trials_for_review:
            self.gui.analyze_trial_frame.remove_q_result.config(
                    state=tk.ACTIVE)

        self.gui.notebook.select(1)

        analysis_frame = self.gui.analyze_trial_frame
        canvas = analysis_frame.plot_canvas.get_tk_widget()
        if analysis_frame.rect is None:
            analysis_frame.rect = canvas.create_rectangle(
                    self.roi[0], self.roi[1], self.roi[2], self.roi[3],
                    fill='', outline='white', tag='rect')
            self.root.update_idletasks()
            for index in range(len(analysis_frame.roi_corners)):
                if index == 0:
                    x_coord = self.roi[0]
                    y_coord = self.roi[1]
                elif index == 1:
                    x_coord = self.roi[2]
                    y_coord = self.roi[1]
                elif index == 2:
                    x_coord = self.roi[2]
                    y_coord = self.roi[3]
                elif index == 3:
                    x_coord = self.roi[0]
                    y_coord = self.roi[3]

                analysis_frame.roi_corners[index] = canvas.create_oval(
                    x_coord - analysis_frame.drag_btn_size,
                    y_coord - analysis_frame.drag_btn_size,
                    x_coord + analysis_frame.drag_btn_size,
                    y_coord + analysis_frame.drag_btn_size,
                    fill='red', tag='corner')
            canvas.tag_bind('corner', "<Any-Enter>", self.mouseEnter)
            canvas.tag_bind('corner', "<Any-Leave>", self.mouseLeave)
            canvas.tag_raise('rect')
            canvas.tag_raise('corner')

        # Check for brightfield image to process
        if self.trial.brightfield_file is not None:
            self.bf_image = pims.open(str(self.trial.brightfield_file))
            bf_image_array = np.asarray(self.bf_image)
            self.bf_image_array = bf_image_array.squeeze()
            self.gui.create_bf_frame(
                    self.bf_image_array.shape[0],
                    self.bf_image_array.shape[1])
            self.gui.notebook.pack(expand=1, fill=tk.BOTH)

#            self.gui.bf_image_frame.create_fig(
#                    self.bf_image_array.shape[0] / self.gui.dpi,
#                    self.bf_image_array.shape[1] / self.gui.dpi)
#            self.gui.bf_image_frame.fig = mpl.figure.Figure(figsize=(
#                    self.bf_image_array.shape[0] / self.gui.dpi,
#                    self.bf_image_array.shape[1] / self.gui.dpi))
#            self.gui.bf_image_frame.ax.clear()
#            self.gui.bf_image_frame.ax.imshow(self.bf_image_array,
#                                              interpolation='none')
            self.gui.bf_image_frame.fig.figimage(self.bf_image_array)
            self.gui.bf_image_frame.ax.axis('off')
            self.gui.bf_image_frame.plot_canvas.draw()

            vulva_dir = self.trial.metadata['vulva_orientation']
            self.gui.bf_image_frame.vulva_orientation.config(
                    text=('Worm vulva oriented ' + vulva_dir))
            trn_name = self.trial.metadata['neuron']
            self.gui.bf_image_frame.neuron_label.config(
                    text=(trn_name + ' stimulated in this experiment'))

            if trn_name == 'AVM' or trn_name == 'PVM':
                self.gui.bf_image_frame.actuation_side = vulva_dir
            else:
                if trn_name == 'ALM' and vulva_dir == 'East':
                    self.gui.bf_image_frame.actuation_side = 'West'
                elif trn_name == 'ALM' and vulva_dir == 'West':
                    self.gui.bf_image_frame.actuation_side = 'East'
            self.gui.bf_image_frame.actuator_location_label.config(
                    text=('Used actuator to the ' +
                          self.gui.bf_image_frame.actuation_side))
            if self.gui.bf_image_frame.actuation_side == 'West':
                print('found actuator to the west')
                self.gui.bf_image_frame.actuator_dir_sign = -1
            elif self.gui.bf_image_frame.actuation_side == 'East':
                print('found actuator to the east')
                self.gui.bf_image_frame.actuator_dir_sign = 1
            else:
                raise ValueError('Actuator should be to the East or West')

            # bindings for Brightfield image analysis frame
            self.gui.bf_image_frame.plot_canvas._tkcanvas.bind(
                    "<Enter>", self._bound_to_mousewheel)
            self.gui.bf_image_frame.plot_canvas._tkcanvas.bind(
                    "<Leave>", self._unbound_to_mousewheel)
            self.gui.bf_image_frame.save_actuator_loc_btn.bind(
                    "<ButtonRelease-1>", func=self.save_actuator_bounds)
            self.gui.bf_image_frame.plot_canvas.draw()
            self.root.update()
            self.gui.bf_image_frame.plot_canvas.draw()

        finish_time = time.time()
        print('Loaded file in ' + str(round(finish_time - start_time)) +
              ' seconds.')

    def find_mitos_one_stack(self, event=None):
        start_time = time.time()

        # Gather parameters
        analysis_frame = self.gui.analyze_trial_frame
        gaussian_width = int(analysis_frame.gaussian_blur_width.get())
        particle_z_diameter = int(analysis_frame.z_diameter_selector.get())
        particle_xy_diameter = int(analysis_frame.xy_diameter_selector.get())
        brightness_percentile = int(
                analysis_frame.brightness_percentile_selector.get())
        min_particle_mass = int(analysis_frame.min_mass_selector.get())
        bottom_slice = int(analysis_frame.btm_slice_selector.get())
        top_slice = int(analysis_frame.top_slice_selector.get())
        time_point = int(analysis_frame.timepoint_selector.get()) - 1

        self.trial.mito_candidates = self.trial.test_parameters(
                images_ndarray=self.trial.image_array,
                roi=self.roi,
                gaussian_width=gaussian_width,
                particle_z_diameter=particle_z_diameter,
                particle_xy_diameter=particle_xy_diameter,
                brightness_percentile=brightness_percentile,
                min_particle_mass=min_particle_mass,
                bottom_slice=bottom_slice,
                top_slice=top_slice,
                time_point=time_point)

        analysis_frame.max_proj_checkbox.state(['selected'])
        analysis_frame.plot_labels_drop.set('Mitos from param test')
        self.update_inspection_image()

        finish_time = time.time()
        print('Time to test parameters for locating mitochondria was ' +
              str(round(finish_time - start_time)) + ' seconds.')

    def find_mitos_current_trial(self, event=None):
        start_time = time.time()
        print('Looking for mitochondria in all timepoints...')
        analysis_frame = self.gui.analyze_trial_frame
        gaussian_width = int(analysis_frame.gaussian_blur_width.get())
        particle_z_diameter = int(analysis_frame.z_diameter_selector.get())
        particle_xy_diameter = int(analysis_frame.xy_diameter_selector.get())
        brightness_percentile = int(
                analysis_frame.brightness_percentile_selector.get())
        min_particle_mass = int(analysis_frame.min_mass_selector.get())
        bottom_slice = int(analysis_frame.btm_slice_selector.get())
        top_slice = int(analysis_frame.top_slice_selector.get())
        tracking_seach_radius = int(
                analysis_frame.linking_radius_selector.get())
        last_timepoint = int(
                analysis_frame.last_time_selector.get())
        notes = analysis_frame.notes_entry.get()

        self.trial.run_batch(
                images_ndarray=self.trial.image_array,
                roi=self.roi,
                gaussian_width=gaussian_width,
                particle_z_diameter=particle_z_diameter,
                particle_xy_diameter=particle_xy_diameter,
                brightness_percentile=brightness_percentile,
                min_particle_mass=min_particle_mass,
                bottom_slice=bottom_slice,
                top_slice=top_slice,
                tracking_seach_radius=tracking_seach_radius,
                last_timepoint=last_timepoint,
                notes=notes)

        analysis_frame.max_proj_checkbox.state(['selected'])
        analysis_frame.plot_labels_drop.set('Plot trajectories')
        self.update_inspection_image()

        done_time = time.time()
        print('Done running file. Batch find took ' +
              str(round(done_time - start_time)) + ' seconds. ')

    def run_multiple_files(self, event=None):
        data_dir = pathlib.Path(self.file_paths['analysis_dir'])
        the_queue = data_dir.joinpath('analysis_queue.yaml')
        for i in range(len(self.file_list)):
            try:
                cur_file = pathlib.Path(self.file_list[i][0])
                exp_id = cur_file.stem
                batch_param_history_file = data_dir.joinpath(
                        exp_id, 'trackpyBatchParamsHistory.yaml')

                with open(batch_param_history_file, 'r') as param_file:
                    entire_history = yaml.safe_load_all(param_file)
                    params = None
                    for params in entire_history:
                        pass  # pass until we get to most recent
            except FileNotFoundError:
                params = self.trial.default_test_params

            param_dict = {'experiment_id': exp_id,
                          'gaussian_width': params['gaussian_width'],
                          'particle_z_diameter': params[
                                  'particle_z_diameter'],
                          'particle_xy_diameter': params[
                                  'particle_xy_diameter'],
                          'brightness_percentile': params[
                                  'brightness_percentile'],
                          'min_particle_mass': params['min_particle_mass'],
                          'bottom_slice': params['bottom_slice'],
                          'top_slice': params['top_slice'],
                          'tracking_seach_radius': params[
                                  'tracking_seach_radius'],
                          'last_timepoint': params['last_timepoint'],
                          'notes': 'Run with defaults'}

            new_queue = []
            overwrite_flag = False
            with open(the_queue, 'r') as queue_file:
                entire_queue = yaml.safe_load_all(queue_file)
                for queue_member in entire_queue:
                    if (queue_member['experiment_id'] == exp_id):
                        new_queue.append(param_dict)
                        overwrite_flag = True
                    else:
                        new_queue.append(queue_member)
                if overwrite_flag is False:
                    new_queue.append(param_dict)

            with open(the_queue, 'w') as output_file:
                    yaml.dump_all(new_queue, output_file,
                                  default_flow_style=False,
                                  explicit_start=True)

#        fr = self.gui.file_load_frame
#        overwrite_metadata = fr.overwrite_metadata_box.instate(['selected'])
#        print('Running lots of files...')
#        for i in range(len(self.file_list)):
#            print('Loading file ', self.file_list[i][0])
#            self.trial.load_trial(self.file_list[i][0],
#                                  load_images=True,
#                                  overwrite_metadata=overwrite_metadata)
#            if 'analysis_status' not in self.trial.metadata:
#                self.trial.metadata['analysis_status'] = 'Testing parameters'
#                self.trial.write_metadata_to_yaml(self.trial.metadata)
#
#            params = self.trial.latest_test_params
#
#            print('Looking for particles...')
#
#            try:
#                self.roi
#            except AttributeError:
#                self.roi = [0,
#                            0,
#                            self.trial.image_array.shape[3],
#                            self.trial.image_array.shape[2]]
#
#            self.trial.run_batch(
#                images_ndarray=self.trial.image_array,
#                roi=self.roi,
#                gaussian_width=params['gaussian_width'],
#                particle_z_diameter=params['particle_z_diameter'],
#                particle_xy_diameter=params['particle_xy_diameter'],
#                brightness_percentile=params['brightness_percentile'],
#                min_particle_mass=params['min_particle_mass'],
#                bottom_slice=params['bottom_slice'],
#                top_slice=params['top_slice'],
#                tracking_seach_radius=params['tracking_seach_radius'],
#                last_timepoint=params['last_timepoint'],
#                notes='none')
#
#            previous_status = self.trial.metadata['analysis_status']
#            if (previous_status == 'No metadata.yaml file' or
#                    previous_status == 'No analysis status yet' or
#                    previous_status == 'Not started'):
#                self.trial.metadata['analysis_status'] = 'Testing parameters'
#                self.trial.write_metadata_to_yaml(self.trial.metadata)
#
#            self.trial = ssn_trial.StrainPropagationTrial()  # clear trial var

    def link_existing_particles(self, event=None):
        """Link previously found particles into trajectories"""
        start_time = time.time()
        analysis_frame = self.gui.analyze_trial_frame
        tracking_seach_radius = int(
                analysis_frame.linking_radius_selector.get())
        last_timepoint = int(
                analysis_frame.last_time_selector.get())

        self.trial.link_mitos(tracking_seach_radius=tracking_seach_radius,
                              last_timepoint=last_timepoint)

        link_done_time = time.time()
        print('Done linking trial file. Linking and filtering took ' +
              str(round(link_done_time - start_time)) + ' seconds.')

    def calculate_strain(self, event=None):
        """Calculate strain between mitochondria as a function of time"""
        self.trial.calculate_strain()
        self.gui.plot_results_frame.plot_strain_one_trial(
                self.trial.strain, self.trial.ycoords_for_strain)
        self.gui.notebook.select(3)

    def add_trial_to_queue(self, event=None):
        """Add trial and analysis parameters to queue for running later"""
        queue_location = self.file_paths['analysis_dir']

        the_queue = queue_location + 'analysis_queue.yaml'

        analysis_frame = self.gui.analyze_trial_frame
        gaussian_width = int(analysis_frame.gaussian_blur_width.get())
        particle_z_diameter = int(analysis_frame.z_diameter_selector.get())
        particle_xy_diameter = int(analysis_frame.xy_diameter_selector.get())
        brightness_percentile = int(
                analysis_frame.brightness_percentile_selector.get())
        min_particle_mass = int(analysis_frame.min_mass_selector.get())
        bottom_slice = int(analysis_frame.btm_slice_selector.get())
        top_slice = int(analysis_frame.top_slice_selector.get())
        tracking_seach_radius = int(
                analysis_frame.linking_radius_selector.get())
        last_timepoint = int(
                analysis_frame.last_time_selector.get())
        notes = analysis_frame.notes_entry.get()

        param_dict = {'experiment_id': self.trial.experiment_id,
                      'roi': self.roi,
                      'gaussian_width': gaussian_width,
                      'particle_z_diameter': particle_z_diameter,
                      'particle_xy_diameter': particle_xy_diameter,
                      'brightness_percentile': brightness_percentile,
                      'min_particle_mass': min_particle_mass,
                      'bottom_slice': bottom_slice,
                      'top_slice': top_slice,
                      'tracking_seach_radius': tracking_seach_radius,
                      'last_timepoint': last_timepoint,
                      'notes': notes}

        new_queue = []
        overwrite_flag = False
        with open(the_queue, 'r') as queue_file:
            entire_queue = yaml.safe_load_all(queue_file)
            for queue_member in entire_queue:
                if (queue_member['experiment_id'] == self.trial.experiment_id):
                    new_queue.append(param_dict)
                    overwrite_flag = True
                else:
                    new_queue.append(queue_member)
            if overwrite_flag is False:
                new_queue.append(param_dict)

        with open(the_queue, 'w') as output_file:
                yaml.dump_all(new_queue, output_file, explicit_start=True)

        try:
            self.remove_from_review_queue(self)
        except Exception as err:
            raise(err)

    def run_queue(self, event=None):
        queue_location = self.file_paths['analysis_dir']
        the_queue = queue_location + 'analysis_queue.yaml'

        with open(the_queue, 'r') as queue_file:
            entire_queue = yaml.safe_load_all(queue_file)
            queue_length = len(list(entire_queue))
        print('Running queue with', queue_length, 'items.')
        while queue_length > 0:
            self.run_queue_item(queue_location)

            # update queue length variable
            with open(the_queue, 'r') as queue_file:
                entire_queue = yaml.safe_load_all(queue_file)
                queue_length = len(list(entire_queue))

        print('Queue is empty.')

    def run_queue_item(self, queue_location):
        """Runs the analysis on the first trial in the queue"""
        start_time = time.time()
        data_location = self.file_paths['data_dir']

        the_queue = queue_location + 'analysis_queue.yaml'
        queue_result_location = queue_location + 'review_queue.yaml'

        # Get first file from queue yaml
        with open(the_queue, 'r') as queue_file:
            entire_queue = yaml.safe_load_all(queue_file)
            params = next(entire_queue)  # get first in line

        # Load file
        full_filename = glob.glob(data_location + '*/' +
                                  params['experiment_id'] + '.nd2')
        print('Running file', full_filename[0])
        self.trial = ssn_trial.StrainPropagationTrial()  # clear trial var
        self.trial.load_trial(full_filename[0],
                              load_images=True, overwrite_metadata=False)

        if 'notes' not in params:
            params['notes'] = 'none'
        if 'roi' not in params:
            params['roi'] = [0,
                             0,
                             self.trial.image_array.shape[3],
                             self.trial.image_array.shape[2]]

        self.trial.run_batch(
                images_ndarray=self.trial.image_array,
                roi=params['roi'],
                gaussian_width=params['gaussian_width'],
                particle_z_diameter=params['particle_z_diameter'],
                particle_xy_diameter=params['particle_xy_diameter'],
                brightness_percentile=params['brightness_percentile'],
                min_particle_mass=params['min_particle_mass'],
                bottom_slice=params['bottom_slice'],
                top_slice=params['top_slice'],
                tracking_seach_radius=params['tracking_seach_radius'],
                last_timepoint=params['last_timepoint'],
                notes=params['notes'])

        # Update analysis status in metadata
        previous_status = self.trial.metadata['analysis_status']
        if (previous_status == 'No metadata.yaml file' or
                previous_status == 'No analysis status yet' or
                previous_status == 'Not started'):
            self.trial.metadata['analysis_status'] = 'Testing parameters'
            self.trial.write_metadata_to_yaml(self.trial.metadata)

        # Add item to list of completed queue items
        with open(queue_result_location, 'a') as review_queue:
            yaml.safe_dump(params, review_queue, explicit_start=True)

        # Remove first trial in queue, since we're done with it
        with open(the_queue, 'r') as queue_file:
            old_queue = yaml.safe_load_all(queue_file)
            new_queue = [item for item in old_queue
                         if item['experiment_id'] != self.trial.experiment_id]

        with open(the_queue, 'w') as queue_file:
            yaml.dump_all(new_queue, queue_file, explicit_start=True)

        done_time = time.time()
        print('Finished file ' + full_filename[0] + ' in ' +
              str(round(done_time - start_time)) + ' seconds. ')

    def remove_from_review_queue(self, event=None):
        """Remove trial from review list that has results from queue"""

        review_q = self.gui.queue_frame.queue_location + 'review_queue.yaml'
        trials_for_review = self.gui.file_load_frame.trials_for_review
        if self.trial.metadata['Experiment_id'] in trials_for_review:
            # Remove first trial in queue, since we're done with it
            with open(review_q, 'r') as queue_file:
                old_queue = yaml.safe_load_all(queue_file)
                new_q = [item for item in old_queue
                         if item['experiment_id'] != self.trial.experiment_id]

            with open(review_q, 'w') as queue_file:
                yaml.dump_all(new_q, queue_file, explicit_start=True)

    def update_status(self, event=None):
        new_status = self.gui.analyze_trial_frame.status_dropdown.get()
        self.trial.metadata['analysis_status'] = new_status
        self.trial.write_metadata_to_yaml(self.trial.metadata)
        self.gui.file_load_frame.file_tree.pack_forget()
        self.gui.file_load_frame.update_file_tree()
        self.gui.root.update()

    def save_actuator_bounds(self, event=None):
        # Save actuator bounds to metadata file for later use
        bf_frame = self.gui.bf_image_frame
        self.trial.metadata['actuator_corners'] = bf_frame.actuator_bounds
        self.trial.metadata['actuator_center'] = [
                float(bf_frame.actuator_center[0]),
                float(bf_frame.actuator_center[1])]
        self.trial.metadata['actuator_thickness'] = bf_frame.actuator_thickness
        self.trial.write_metadata_to_yaml(self.trial.metadata)

    def plot_existing_strain(self, event=None):
        """"Plots strain of an existing trial"""
        plot_tab = self.gui.plot_results_frame
        tree = plot_tab.plot_strain_tree
        selection = tree.selection()
        for this_item in selection:
            info = tree.item(this_item)
            exp_id = info['text'][0:-4]

        data_location = (self.file_paths['analysis_dir'] + exp_id)
        self.trial.metadata_file_path = data_location + '/metadata.yaml'

        strain_file = data_location + '/strain.yaml'
        with open(strain_file, 'r') as yamlfile:
            strain_results = yaml.safe_load(yamlfile)

        strain = strain_results['strain']
        ycoords = strain_results['ycoords']

        if self.trial.metadata is None:
            metadata = self.trial.load_metadata_from_yaml()
        else:
            metadata = self.trial.metadata

        if event.widget == plot_tab.plot_strain_one_trial_button:
            self.gui.plot_results_frame.plot_strain_one_trial(strain, ycoords)
        elif event.widget == plot_tab.plot_strain_by_actuation_btn:
            self.gui.plot_results_frame.plot_strain_by_actuation(
                strain, ycoords, metadata['pressure_kPa'])

    def plot_xz_displacements(self, event=None):
        """Plots displacement of mitos projected on xz plane"""
        plot_tab = self.gui.plot_results_frame
        tree = plot_tab.plot_strain_tree
        selection = tree.selection()
        for this_item in selection:
            info = tree.item(this_item)
            exp_id = info['text'][0:-4]

        data_location = (self.file_paths['analysis_dir'] + exp_id)
        self.trial.metadata_file_path = data_location + '/metadata.yaml'
        if self.trial.metadata is None:
            metadata = self.trial.load_metadata_from_yaml()
        else:
            metadata = self.trial.metadata

        self.trial.batch_data_file = (data_location +
                                      '/trackpyBatchResults.yaml')
        with open(self.trial.batch_data_file, 'r') as yamlfile:
                linked_mitos_dict = yaml.safe_load(yamlfile)
                self.trial.linked_mitos = pd.DataFrame.from_dict(
                        linked_mitos_dict, orient='index')
        mitos_df = self.trial.linked_mitos.copy(deep=True)
        num_trajectories = mitos_df['particle'].nunique()
        num_frames = mitos_df['frame'].nunique()
        xz_distances = np.empty([num_frames, num_trajectories])
        ycoords = np.empty([num_frames, num_trajectories])
        for stack in range(num_frames - 1):
            # sort mitochondria in this stack by y values
            current_stack = mitos_df.loc[mitos_df['frame'] == stack]
            current_stack.sort_values(['y'], inplace=True)
            current_stack.reset_index(inplace=True, drop=True)

            next_stack = mitos_df.loc[mitos_df['frame'] == stack + 1]
            next_stack.sort_values(['y'], inplace=True)
            next_stack.reset_index(inplace=True, drop=True)

            for particle in range(num_trajectories):
                # calculate pairwise distances
                mito1 = current_stack.loc[particle, ['x', 'z']].values
                mito2 = next_stack.loc[particle, ['x', 'z']].values
                xz_distances[stack, particle] = spatial.distance.euclidean(
                        mito1, mito2)
                ycoords[stack, particle] = current_stack.loc[
                        particle, ['y']].values
                ycoords[stack + 1, particle] = next_stack.loc[
                        particle, ['y']].values

        plot_tab.plot_xz_displacements(xz_distances, ycoords,
                                       metadata['pressure_kPa'])

    def on_file_selection_changed(self, event):
        """This function keeps track of which files are selected for
        analysis. If more than one files are selected, the Run Batch option
        is available, otherwise only the Load Trial button is active.
        """
        file_tree = self.gui.file_load_frame.file_tree
        file_load_frame = self.gui.file_load_frame
        self.file_list = []
        selection = file_tree.selection()
        for this_item in selection:
            info = file_tree.item(this_item)
            self.file_list.append(info['tags'])
        if len(self.file_list) > 1:
            file_load_frame.load_trial_button.config(state=tk.DISABLED)
            file_load_frame.run_multiple_files_button.config(state=tk.NORMAL)
        elif len(self.file_list) == 1 and info['tags'] != ['day']:
            file_load_frame.load_trial_button.config(state=tk.NORMAL)
            file_load_frame.run_multiple_files_button.config(state=tk.DISABLED)

    def spinbox_delay_then_update_image(self, event=None):
        self.gui.analyze_trial_frame.after(1, self.update_inspection_image)

    def update_inspection_image(self, event=None):
        # TODO: arrange roi lines on top of image
        # get parameters
        current_frame = self.gui.analyze_trial_frame
        scroll_pos = current_frame.scrollbar.get()
        plot_mitos_status = current_frame.plot_labels_drop.get()
        max_proj_checkbox = current_frame.max_proj_checkbox.instate(
                ['selected'])
        particle_label_checkbox = current_frame.part_label_checkbox.instate(
                ['selected'])
        selected_slice = int(current_frame.slice_selector.get()) - 1
        selected_timepoint = int(current_frame.timepoint_selector.get()) - 1
        min_pixel = int(current_frame.min_pixel_disp.get())
        max_pixel = int(current_frame.max_pixel_disp.get())
        current_frame.ax.clear()
        df_for_plot = None
        df_for_inspection = None
        connect_points = False

        # get image that we're going to show
        if self.trial.image_array is not None:
            if max_proj_checkbox is True:  # max projection
                stack = self.trial.image_array[selected_timepoint]
                image_to_display = np.amax(stack, 0)  # collapse z axis
                image_to_display = image_to_display.squeeze()
            else:  # single slice
                image_to_display = self.trial.image_array[
                        selected_timepoint, selected_slice]
        else:
            # get stored image if available
            try:
                if plot_mitos_status == 'Unlinked mitos for this stack':
                    one_stack_fig = self.trial.analyzed_data_location.joinpath(
                            'diag_images/stack_' +
                            str(selected_timepoint) + '_fig.png')
                    image_to_display = plt.imread(str(one_stack_fig))
                else:
                    traj_fig_file = self.trial.analyzed_data_location.joinpath(
                            'diag_images/trajectory_fig.png')
                    image_to_display = plt.imread(str(traj_fig_file))
            except FileNotFoundError:
                pass

        # get particle positions to plot, if any
        if plot_mitos_status == 'Plot trajectories':
            df_for_plot = self.trial.linked_mitos[
                    ['particle', 'x', 'y']].copy()
            df_for_inspection = self.trial.linked_mitos.copy()
            connect_points = True
        elif plot_mitos_status == 'Linked mitos for this stack':
            df_for_plot = self.trial.linked_mitos.loc[
                    self.trial.linked_mitos[
                            'frame'] == selected_timepoint][
                            ['particle', 'x', 'y']].copy()
            df_for_inspection = self.trial.linked_mitos.loc[
                    self.trial.linked_mitos[
                            'frame'] == selected_timepoint].copy()
        elif plot_mitos_status == 'Unlinked mitos for this stack':
            df_for_plot = self.trial.mitos_from_batch.loc[
                    self.trial.mitos_from_batch[
                            'frame'] == selected_timepoint][
                            ['x', 'y']].copy()
            df_for_inspection = self.trial.mitos_from_batch.loc[
                    self.trial.mitos_from_batch[
                            'frame'] == selected_timepoint].copy()
        elif plot_mitos_status == 'Mitos from param test':
            df_for_plot = self.trial.mito_candidates[['x', 'y']].copy()
            df_for_inspection = self.trial.mito_candidates.copy()
        elif plot_mitos_status == 'No overlay':
            pass

        # show image
        current_frame.update_image(image=image_to_display,
                                   plot_data=df_for_plot,
                                   min_pixel=min_pixel,
                                   max_pixel=max_pixel,
                                   connect_points_over_time=connect_points,
                                   show_text_labels=particle_label_checkbox)
        # roi: [271, 0, 339, 1054]
        #        controller.trial.image_array.shape
        # (11, 67, 1070, 602)

        if self.roi is not None:
            side_stack = self.trial.image_array[selected_timepoint,
                                                :, :, self.roi[0]:self.roi[2]]
            image_to_display = np.amax(side_stack, 2).transpose()
            current_frame.update_side_view(
                    image=image_to_display,
                    plot_data=df_for_plot,
                    min_pixel=min_pixel,
                    max_pixel=max_pixel,
                    connect_points_over_time=connect_points,
                    show_text_labels=particle_label_checkbox)
            if max_proj_checkbox is False:
                side_im = current_frame.side_canvas.get_tk_widget()
                try:
                    side_im.delete('side_line')
                except NameError:
                    pass
                side_im.create_line(selected_slice, 0,
                                    selected_slice, 1200,
                                    fill="white",
                                    dash=(4, 4),
                                    tags='side_line')

        current_frame.plot_canvas.get_tk_widget().yview_moveto(scroll_pos[0])
        current_frame.side_canvas.get_tk_widget().yview_moveto(scroll_pos[0])

        if df_for_inspection is not None:
            df_for_inspection.sort_values('y', inplace=True)
            columns = ['x', 'y', 'z', 'mass',
                       'signal', 'raw_mass', 'size_x', 'size_y',
                       'size_z', 'ecc', 'ep_x', 'ep_y', 'ep_z']
            if 'particle' in df_for_inspection:
                columns.insert(4, 'particle')
            if 'frame' in df_for_inspection:
                columns.insert(4, 'frame')

            df_for_inspection = df_for_inspection[columns]

#            current_frame.dataframe_widget.show()
            current_frame.dataframe_widget.updateModel(
                    TableModel(df_for_inspection))

            current_frame.dataframe_widget.redraw()

        # reapply bindings again for scrolling
        self.gui.analyze_trial_frame.plot_canvas._tkcanvas.bind(
                "<ButtonPress-1>", self.on_click_image)
        self.gui.analyze_trial_frame.plot_canvas._tkcanvas.bind(
                "<B1-Motion>", self.on_drag_roi)
        self.gui.analyze_trial_frame.plot_canvas._tkcanvas.bind(
                "<ButtonRelease-1>", self.on_click_image_release)
        self.gui.analyze_trial_frame.plot_canvas._tkcanvas.bind(
                "<Enter>", self._bound_to_mousewheel)
        self.gui.analyze_trial_frame.plot_canvas._tkcanvas.bind(
                "<Leave>", self._unbound_to_mousewheel)

    def update_histogram(self, event=None):
        current_frame = self.gui.analyze_trial_frame
        nb = current_frame.analysis_notebook
        tab = nb.tab(nb.select(), "text")
        if tab == 'Histogram':
            min_pixel = int(current_frame.min_pixel_disp.get())
            max_pixel = int(current_frame.max_pixel_disp.get())
            max_proj_checkbox = current_frame.max_proj_checkbox.instate(
                    ['selected'])
            selected_timepoint = int(
                    current_frame.timepoint_selector.get()) - 1
            selected_slice = int(current_frame.slice_selector.get()) - 1
            current_frame.hist_ax.clear()

            if max_proj_checkbox is True:  # max projection
                stack = self.trial.image_array[selected_timepoint]
                image_to_display = np.amax(stack, 0)  # collapse z axis
                image_to_display = image_to_display.squeeze()
            else:  # single slice
                image_to_display = self.trial.image_array[
                            selected_timepoint, selected_slice]

            bins = max_pixel - min_pixel + 1
            bin_list = list(range(min_pixel, max_pixel + 1))
            hist_array = fast_histogram.histogram1d(image_to_display,
                                                    bins=bins,
                                                    range=(min_pixel,
                                                           max_pixel))

            current_frame.hist_ax.bar(bin_list, hist_array, width=1)
            current_frame.hist_ax.set_yscale('log')
            current_frame.histogram_canvas.draw()

    def on_arrow_key(self, event=None):
        nb = self.gui.notebook
        tab = nb.tab(nb.select(), "text")
        if tab == 'Inspect Images':
            current_slice = self.gui.analyze_trial_frame.slice_selector.get()
            if event.keysym == 'Down':
                new_slice = int(current_slice) - 1
            if event.keysym == 'Up':
                new_slice = int(current_slice) + 1
            self.gui.analyze_trial_frame.slice_selector.delete(0, 'end')
            self.gui.analyze_trial_frame.slice_selector.insert(0, new_slice)
            self.update_inspection_image(event=event.keysym)

    def on_click_image(self, event=None):
        canvas = event.widget  # analysis_frame.plot_canvas.get_tk_widget()
        init_size = 100
        cur_x = canvas.canvasx(event.x)
        cur_y = canvas.canvasy(event.y)
        analysis_frame = self.gui.analyze_trial_frame
        self.roi = [cur_x,
                    cur_y,
                    cur_x + init_size,
                    cur_y + init_size]

        # create rectangle if not yet existing
        if analysis_frame.rect is None:
            analysis_frame.rect = canvas.create_rectangle(
                    cur_x, cur_y, cur_x + init_size, cur_y + init_size,
                    fill='', outline='white')
            for index in range(len(analysis_frame.roi_corners)):
                if index == 0:
                    x_shift = 0
                    y_shift = 0
                elif index == 1:
                    x_shift = init_size
                    y_shift = 0
                elif index == 2:
                    x_shift = init_size
                    y_shift = init_size
                elif index == 3:
                    x_shift = 0
                    y_shift = init_size

                analysis_frame.roi_corners[index] = canvas.create_oval(
                    cur_x - analysis_frame.drag_btn_size + x_shift,
                    cur_y - analysis_frame.drag_btn_size + y_shift,
                    cur_x + analysis_frame.drag_btn_size + x_shift,
                    cur_y + analysis_frame.drag_btn_size + y_shift,
                    fill='red', tag='corner')

            canvas.tag_bind('corner', "<Any-Enter>", self.mouseEnter)
            canvas.tag_bind('corner', "<Any-Leave>", self.mouseLeave)

        analysis_frame.last_coords = (cur_x, cur_y)

    def on_drag_roi(self, event=None):
        analysis_frame = self.gui.analyze_trial_frame
        canvas = event.widget
        cur_x = canvas.canvasx(event.x)
        cur_y = canvas.canvasy(event.y)

        if canvas.type(tk.CURRENT) == 'oval':  # if corner selected
            current_item_coords = canvas.coords(tk.CURRENT)
            new_bounds = [cur_x - analysis_frame.drag_btn_size,
                          cur_y - analysis_frame.drag_btn_size,
                          cur_x + analysis_frame.drag_btn_size,
                          cur_y + analysis_frame.drag_btn_size]
            for i in range(len(analysis_frame.roi_corners)):
                corner_coords = canvas.coords(
                        analysis_frame.roi_corners[i])
                # Find out which corner is moving
                compare_coords = [x == y for (x, y) in zip(
                        current_item_coords, corner_coords)]
                # Only keep old coords if we're looking at opposite corner
                new_coords = [new_crd if move_bool else old_crd
                              for old_crd, new_crd, move_bool
                              in zip(corner_coords,
                                     new_bounds,
                                     compare_coords)]
                if sum(compare_coords) == 0:
                    opposite_corner_coords = new_coords
                canvas.coords(analysis_frame.roi_corners[i], new_coords)

            new_rect_coords = [sum(new_bounds[0::2]) / 2,
                               sum(new_bounds[1::2]) / 2,
                               sum(opposite_corner_coords[0::2]) / 2,
                               sum(opposite_corner_coords[1::2]) / 2]
            canvas.coords(analysis_frame.rect, new_rect_coords)

            self.roi = new_rect_coords

    def on_click_image_release(self, event=None):
        # limit ROI to values that make sense
        for i in range(len(self.roi)):
            if i % 2 == 0:
                # x value
                max_val = self.trial.image_array.shape[3]
            else:
                # y value
                max_val = self.trial.image_array.shape[2]
#            if tab_index == 1:
            self.roi[i] = np.clip(self.roi[i], 0, max_val)

            # make sure lower values are listed first
            xmin = int(min([self.roi[0], self.roi[2]]))
            xmax = int(max([self.roi[0], self.roi[2]]))
            ymin = int(min([self.roi[1], self.roi[3]]))
            ymax = int(max([self.roi[1], self.roi[3]]))
            self.roi = [xmin, ymin, xmax, ymax]

            print('Selected ROI: ', self.roi)

    def mouseEnter(self, event):
        analysis_frame = self.gui.analyze_trial_frame
        canvas = event.widget  # analysis_frame.plot_canvas.get_tk_widget()
        canvas.itemconfig(tk.CURRENT, fill="green")
        analysis_frame.selected_item = tk.CURRENT

    def mouseLeave(self, event):
        analysis_frame = self.gui.analyze_trial_frame
        canvas = event.widget  # analysis_frame.plot_canvas.get_tk_widget()
        canvas.itemconfig(tk.CURRENT, fill="red")
        analysis_frame.selected_item = None

    def clear_roi(self, event=None):
        analysis_frame = self.gui.analyze_trial_frame
        canvas = analysis_frame.plot_canvas.get_tk_widget()
        canvas.delete('corner')
        canvas.delete(analysis_frame.rect)

        analysis_frame.rect = None
        analysis_frame.roi_corners = [None, None, None, None]
        analysis_frame.roi = [None, None, None, None]

        self.roi = [0,
                    0,
                    self.trial.image_array.shape[3],
                    self.trial.image_array.shape[2]]

        print('Selected ROI: ', self.roi)
        self.update_inspection_image()

    def _bound_to_mousewheel(self, event):
        event.widget.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbound_to_mousewheel(self, event):
        event.widget.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        event.widget.yview_scroll(-1*(event.delta), 'units')
        frame = self.gui.analyze_trial_frame
        if event.widget == frame.side_canvas.get_tk_widget():
            frame.plot_canvas.get_tk_widget().yview_scroll(
                    -1 * (event.delta), 'units')
        elif event.widget == frame.plot_canvas.get_tk_widget():
            frame.side_canvas.get_tk_widget().yview_scroll(
                    -1 * (event.delta), 'units')

    def get_analysis_progress(self, event=None):
        """Gets the analysis status of all trials and plots their status"""

#        all_statuses_dict = {}
        status_values = self.trial.STATUSES
#        data_dir = self.file_paths['data_dir']
#        data_location = (data_dir + '*/SSN_*.nd2')
#
#        analysis_dir = self.file_paths['analysis_dir']
#        metadata_location = analysis_dir + 'AnalyzedData/'
#
#        # for all subfiles ending in .nd2
#        for nd2_file in glob.glob(data_location):
#            experiment_id = nd2_file[-15:-4]
#            metadata_file = (metadata_location +
#                             experiment_id +
#                             '/metadata.yaml')
#            # try to load metadata
#            if 'bf' not in experiment_id:
#                try:
#                    with open(metadata_file, 'r') as yamlfile:
#                        metadata = yaml.safe_load(yamlfile)
#                        # if metadata exists, get analysis status and store it
#                        all_statuses_dict[experiment_id] = (
#                                metadata['analysis_status'],
#                                metadata['neuron'],
#                                metadata['worm_strain'])
#                except FileNotFoundError:
#                    all_statuses_dict[experiment_id] = (
#                            'No metadata.yaml file', 'N/A', 'N/A')
#                except KeyError:
#                    all_statuses_dict[experiment_id] = (
#                            'No analysis status yet', 'N/A', 'N/A')

        # send all analysis statuses to view module for plotting
        self.gui.plot_results_frame.plot_progress(status_values)

    def _display_last_test_params(self, event=None):
        analysis_frame = self.gui.analyze_trial_frame
        latest_params = self.trial.latest_test_params
        params = {**self.trial.default_test_params, **latest_params}

        analysis_frame.btm_slice_selector.delete(0, 'end')
        analysis_frame.btm_slice_selector.insert(0, params['bottom_slice'])

        analysis_frame.top_slice_selector.delete(0, 'end')
        analysis_frame.top_slice_selector.insert(0, params['top_slice'])

        analysis_frame.gaussian_blur_width.delete(0, 'end')
        if 'noise_size' in params:
            analysis_frame.gaussian_blur_width.insert(0, params['noise_size'])
        else:
            analysis_frame.gaussian_blur_width.insert(0,
                                                      params['gaussian_width'])

        analysis_frame.z_diameter_selector.delete(0, 'end')
        analysis_frame.xy_diameter_selector.delete(0, 'end')
        if 'diameter' in params:
            analysis_frame.z_diameter_selector.insert(
                    0, params['diameter'][0])
            analysis_frame.xy_diameter_selector.insert(
                    0, params['diameter'][1])
        else:
            analysis_frame.z_diameter_selector.insert(
                    0, params['particle_z_diameter'])
            analysis_frame.xy_diameter_selector.insert(
                    0, params['particle_xy_diameter'])

        analysis_frame.brightness_percentile_selector.delete(0, 'end')
        if 'percentile' in params:
            analysis_frame.brightness_percentile_selector.insert(
                    0, params['percentile'])
        else:
            analysis_frame.brightness_percentile_selector.insert(
                    0, params['brightness_percentile'])

        analysis_frame.min_mass_selector.delete(0, 'end')
        if 'minmass' in params:
            analysis_frame.min_mass_selector.insert(
                0, params['minmass'])
        else:
            analysis_frame.min_mass_selector.insert(
                0, params['min_particle_mass'])

        analysis_frame.last_time_selector.delete(0, 'end')
        analysis_frame.last_time_selector.insert(
                0, params['last_timepoint'])

        analysis_frame.linking_radius_selector.delete(0, 'end')
        analysis_frame.linking_radius_selector.insert(
                0, params['tracking_seach_radius'])


if __name__ == '__main__':
    controller = StrainGUIController()
    controller.run()
