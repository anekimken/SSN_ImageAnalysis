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
import glob
import yaml
import matplotlib.pyplot as plt
import fast_histogram
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

        # Instantiate Model
        self.trial = ssn_trial.StrainPropagationTrial()

        if headless is not True:
            # Instantiate View
            self.gui = ssn_view.SSN_analysis_GUI(self.root)

            # Bind UI elements to functions
            # Load trial tab
            self.gui.file_load_frame.load_trial_button.bind(
                    "<ButtonRelease-1>", func=self.load_trial)
            self.gui.file_load_frame.run_multiple_files_button.bind(
                    "<ButtonRelease-1>", func=self.run_multiple_files)
            self.gui.file_load_frame.file_tree.bind(
                    '<<TreeviewSelect>>', func=self.on_file_selection_changed)

            # Analysis frame
            self.gui.analyze_trial_frame.plot_canvas._tkcanvas.bind(
                    "<ButtonPress-1>", self.on_button_press)
            self.gui.analyze_trial_frame.plot_canvas._tkcanvas.bind(
                    "<B1-Motion>", self.on_move_press)
            self.gui.analyze_trial_frame.plot_canvas._tkcanvas.bind(
                    "<ButtonRelease-1>", self.on_button_release)
            self.gui.analyze_trial_frame.plot_canvas._tkcanvas.bind(
                    "<Enter>", self._bound_to_mousewheel)
            self.gui.analyze_trial_frame.plot_canvas._tkcanvas.bind(
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
            self.gui.analyze_trial_frame.min_pixel_disp.bind(
                    "<ButtonRelease-1>",
                    func=self.spinbox_delay_then_update_image)
            self.gui.analyze_trial_frame.max_pixel_disp.bind(
                    "<ButtonRelease-1>",
                    func=self.spinbox_delay_then_update_image)
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

            # Queue frame
            self.gui.queue_frame.run_queue_button.bind(
                    "<ButtonRelease-1>", func=self.run_queue)

            # Plotting frame
            self.gui.plot_results_frame.progress_plot_button.bind(
                    "<ButtonRelease-1>", func=self.get_analysis_progress)
            self.gui.plot_results_frame.plot_strain_one_trial_button.bind(
                    "<ButtonRelease-1>", func=self.plot_existing_strain)
            # TODO: write callback for plot strain one trial button

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

        # load inspection image on inspection tab and related parameters
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

        self.roi = self.trial.latest_test_params['roi']

        self._display_last_test_params()

        # load param history into text field
        # TODO: test what happens if this file doesn't exist yet
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
        # TODO: merge into queue?
        fr = self.gui.file_load_frame
        overwrite_metadata = fr.overwrite_metadata_box.instate(['selected'])
        print('Running lots of files...')
        for i in range(len(self.file_list)):
            print('Loading file ', self.file_list[i][0])
            self.trial.load_trial(self.file_list[i][0],
                                  load_images=True,
                                  overwrite_metadata=overwrite_metadata)
            if 'analysis_status' not in self.trial.metadata:
                self.trial.metadata['analysis_status'] = 'Testing parameters'
                self.trial.write_metadata_to_yaml(self.trial.metadata)

            params = self.trial.latest_test_params

            print('Looking for particles...')

            try:
                self.roi
            except AttributeError:
                self.roi = [0,
                            0,
                            self.trial.image_array.shape[3],
                            self.trial.image_array.shape[2]]

            self.trial.run_batch(
                images_ndarray=self.trial.image_array,
                roi=self.roi,
                gaussian_width=params['gaussian_width'],
                particle_z_diameter=params['particle_z_diameter'],
                particle_xy_diameter=params['particle_xy_diameter'],
                brightness_percentile=params['brightness_percentile'],
                min_particle_mass=params['min_particle_mass'],
                bottom_slice=params['bottom_slice'],
                top_slice=params['top_slice'],
                tracking_seach_radius=params['tracking_seach_radius'],
                last_timepoint=params['last_timepoint'],
                notes='none')

            previous_status = self.trial.metadata['analysis_status']
            if (previous_status == 'No metadata.yaml file' or
                    previous_status == 'No analysis status yet' or
                    previous_status == 'Not started'):
                self.trial.metadata['analysis_status'] = 'Testing parameters'
                self.trial.write_metadata_to_yaml(self.trial.metadata)

            self.trial = ssn_trial.StrainPropagationTrial()  # clear trial var

    def run_queue(self, event=None):
        queue_location = self.gui.queue_frame.queue_location
        the_queue = queue_location + 'analysis_queue.yaml'

        with open(the_queue, 'r') as queue_file:
            entire_queue = yaml.load_all(queue_file)
            queue_length = len(list(entire_queue))
        print('Running queue with', queue_length, 'items.')
        while queue_length > 0:
            self.run_queue_item(queue_location)

            # update queue length variable
            with open(the_queue, 'r') as queue_file:
                entire_queue = yaml.load_all(queue_file)
                queue_length = len(list(entire_queue))

        print('Queue is empty.')

    def run_queue_item(self, queue_location):
        """Runs the analysis on the first trial in the queue"""
        start_time = time.time()
        data_location = '/Users/adam/Documents/SenseOfTouchResearch/SSN_data/'
        the_queue = queue_location + 'analysis_queue.yaml'
        queue_result_location = queue_location + 'review_queue.yaml'

        # Get first file from queue yaml
        with open(the_queue, 'r') as queue_file:
            entire_queue = yaml.load_all(queue_file)
            params = next(entire_queue)  # get first in line

        # Load file
        full_filename = glob.glob(data_location + '*/' +
                                  params['experiment_id'] + '.nd2')
        print('Running file', full_filename[0])
        self.trial = ssn_trial.StrainPropagationTrial()  # clear trial var
        self.trial.load_trial(full_filename[0],
                              load_images=True, overwrite_metadata=False)

        # TODO: add try/except block for trackpy computational errors
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
            old_queue = yaml.load_all(queue_file)
            new_queue = [item for item in old_queue
                         if item['experiment_id'] != self.trial.experiment_id]

        with open(the_queue, 'w') as queue_file:
            yaml.dump_all(new_queue, queue_file, explicit_start=True)

        done_time = time.time()
        print('Finished file ' + full_filename[0] + ' in ' +
              str(round(done_time - start_time)) + ' seconds. ')

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

    def plot_existing_strain(self, event=None):
        """"Plots strain of an existing trial"""
        tree = self.gui.plot_results_frame.plot_strain_tree
        selection = tree.selection()
        for this_item in selection:
            info = tree.item(this_item)
            exp_id = info['text'][0:-4]

        data_location = ('/Users/adam/Documents/SenseOfTouchResearch/'
                         'SSN_ImageAnalysis/AnalyzedData/' + exp_id)
        strain_file = data_location + '/strain.yaml'
        self.trial.metadata_file_path = data_location + '/metadata.yaml'
        with open(strain_file, 'r') as yamlfile:
            strain_results = yaml.safe_load(yamlfile)

        strain = strain_results['strain']
        ycoords = strain_results['ycoords']

        if self.trial.metadata is None:
            metadata = self.trial.load_metadata_from_yaml()
        else:
            metadata = self.trial.metadata

#        self.gui.plot_results_frame.plot_strain_one_trial(strain, ycoords)
        self.gui.plot_results_frame.plot_strain_by_actuation(
                strain, ycoords, metadata['pressure_kPa'])

    def add_trial_to_queue(self, event=None):
        """Add trial and analysis parameters to queue for running later"""

        queue_location = self.gui.queue_frame.queue_location
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
            entire_queue = yaml.load_all(queue_file)
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

    def remove_from_review_queue(self, event=None):
        """Remove trial from review list that has results from queue"""

        review_q = self.gui.queue_frame.queue_location + 'review_queue.yaml'
        trials_for_review = self.gui.file_load_frame.trials_for_review
        if self.trial.metadata['Experiment_id'] in trials_for_review:
            # Remove first trial in queue, since we're done with it
            with open(review_q, 'r') as queue_file:
                old_queue = yaml.load_all(queue_file)
                new_q = [item for item in old_queue
                         if item['experiment_id'] != self.trial.experiment_id]

            with open(review_q, 'w') as queue_file:
                yaml.dump_all(new_q, queue_file, explicit_start=True)

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
        fr = self.gui.file_load_frame
        load_images = fr.load_images_box.instate(['selected'])
        overwrite_metadata = fr.overwrite_metadata_box.instate(['selected'])

        analysis_frame = self.gui.analyze_trial_frame
        plot_mitos_status = analysis_frame.plot_labels_drop.get()
        max_proj_checkbox = analysis_frame.max_proj_checkbox.instate(
                ['selected'])
        particle_label_checkbox = analysis_frame.part_label_checkbox.instate(
                ['selected'])
        selected_slice = int(analysis_frame.slice_selector.get())
        selected_timepoint = int(analysis_frame.timepoint_selector.get()) - 1
        min_pixel = int(analysis_frame.min_pixel_disp.get())
        max_pixel = int(analysis_frame.max_pixel_disp.get())
        analysis_frame.ax.clear()
        analysis_frame.hist_ax.clear()
        df_for_plot = None

        # Get dataframe with data we're going to plot on top of image
        if plot_mitos_status == 'Linked mitos for this stack':
            if (self.trial.linked_mitos is not None and
                    selected_timepoint - 1 <=
                    max(self.trial.linked_mitos['frame'])):
                df_for_plot = self.trial.linked_mitos.loc[
                        self.trial.linked_mitos[
                                'frame'] == selected_timepoint]
                mito_labels = df_for_plot
                particle_marker = 'o'
                traj_line = 'None'
        elif plot_mitos_status == 'Plot trajectories':
            if self.trial.linked_mitos is not None:
                df_for_plot = self.trial.linked_mitos
                mito_labels = df_for_plot
                particle_marker = 'None'
                traj_line = '-'
        elif self.trial.linked_mitos is not None:
            mito_labels = self.trial.linked_mitos.loc[
                            self.trial.linked_mitos[
                                    'frame'] == selected_timepoint]
            particle_marker = 'None'
            traj_line = 'None'
        else:
            mito_labels = None

        # Plot data with text
        try:
            theCount = 0  # ah ah ah ah
            for i in range(max(mito_labels['particle'])):
                this_particle = mito_labels.loc[mito_labels['particle'] == i+1]
                if not this_particle.empty:
                    theCount += 1
                    this_particle.plot(
                            x='x',
                            y='y',
                            ax=analysis_frame.ax,
                            color='#FB8072',
                            marker=particle_marker,
                            linestyle=traj_line)
                    if particle_label_checkbox is True:
                        analysis_frame.ax.text(
                            this_particle['x'].mean() + 15,
                            this_particle['y'].mean(),
                            str(int(this_particle.iloc[0][
                                     'particle'])), color='white')
                    analysis_frame.ax.legend_.remove()
        except ValueError as err:
            if len(self.trial.linked_mitos) == 0:
                raise Exception('No particles were found at all '
                                'time points. Try expanding search '
                                'radius or changing particle finding '
                                'parameters.') from err
            else:
                raise
        except TypeError:
            if df_for_plot is None:
                pass
            else:
                raise

        if plot_mitos_status == 'Unlinked mitos for this stack':
            if (self.trial.mitos_from_batch is not None and
                    selected_timepoint - 1 <=
                    max(self.trial.mitos_from_batch['frame'])):
                df_for_plot = self.trial.mitos_from_batch.loc[
                        self.trial.mitos_from_batch[
                                'frame'] == selected_timepoint]
        elif plot_mitos_status == 'Mitos from param test':
            if self.trial.mito_candidates is not None:
                df_for_plot = self.trial.mito_candidates
        # plot data before linking, so no text here
        if (plot_mitos_status == 'Mitos from param test' or
                plot_mitos_status == 'Unlinked mitos for this stack'):
            df_for_plot.plot(x='x', y='y', ax=analysis_frame.ax,
                             color='#FB8072', marker='o', linestyle='None')

        if df_for_plot is not None:
            self.gui.analyze_trial_frame.dataframe = df_for_plot
            self.gui.analyze_trial_frame.dataframe_widget.updateModel(
                    TableModel(df_for_plot))
            self.gui.analyze_trial_frame.dataframe_widget.redraw()

        if load_images is True or overwrite_metadata is True:
            # get image that we're going to show
            if max_proj_checkbox is True:  # max projection
                stack = self.trial.image_array[selected_timepoint]
                image_to_display = np.amax(stack, 0)  # collapse z axis
                image_to_display = image_to_display.squeeze()
            else:  # single slice
                image_to_display = self.trial.image_array[
                        selected_timepoint, selected_slice]

            analysis_frame.fig.set_size_inches(
                    forward=True,
                    h=image_to_display.shape[0] / self.gui.dpi,
                    w=image_to_display.shape[1] / self.gui.dpi)
            analysis_frame.ax.imshow(image_to_display, interpolation='none',
                                     vmin=min_pixel, vmax=max_pixel)
            analysis_frame.ax.axis('off')
            analysis_frame.plot_canvas.draw()

            min_pixel = int(np.amin(image_to_display))
            max_pixel = int(np.amax(image_to_display))
            bins = max_pixel - min_pixel + 1
            bin_list = list(range(min_pixel, max_pixel+1))
            hist_array = fast_histogram.histogram1d(image_to_display,
                                                    bins=bins,
                                                    range=(min_pixel,
                                                           max_pixel))

            analysis_frame.hist_ax.bar(bin_list, hist_array, width=1)
            analysis_frame.hist_ax.set_yscale('log')
            analysis_frame.histogram_canvas.draw()
        else:
            if plot_mitos_status == 'Unlinked mitos for this stack':
                one_stack_fig = self.trial.analyzed_data_location.joinpath(
                        'diag_images/stack_' +
                        str(selected_timepoint) + '_fig.png')
                self.saved_photo = plt.imread(str(one_stack_fig))
            else:
                traj_fig_file = self.trial.analyzed_data_location.joinpath(
                        'diag_images/trajectory_fig.png')
                self.saved_photo = plt.imread(str(traj_fig_file))
            analysis_frame.ax.imshow(self.saved_photo)
            analysis_frame.plot_canvas.draw()

    def on_button_press(self, event=None):
        analysis_frame = self.gui.analyze_trial_frame
        canvas = analysis_frame.plot_canvas.get_tk_widget()
        init_size = 100
        cur_x = canvas.canvasx(event.x)
        cur_y = canvas.canvasy(event.y)
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

    def on_move_press(self, event=None):
        analysis_frame = self.gui.analyze_trial_frame
        canvas = analysis_frame.plot_canvas.get_tk_widget()
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
                compare_coords = [x == y for (x, y) in zip(
                        current_item_coords, corner_coords)]
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

    def on_button_release(self, event=None):
        # limit ROI to values that make sense
        for i in range(len(self.roi)):
            if i % 2 == 0:
                # x value
                max_val = self.trial.image_array.shape[3]
            else:
                # y value
                max_val = self.trial.image_array.shape[2]
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
        canvas = analysis_frame.plot_canvas.get_tk_widget()
        canvas.itemconfig(tk.CURRENT, fill="green")
        analysis_frame.selected_item = tk.CURRENT

    def mouseLeave(self, event):
        analysis_frame = self.gui.analyze_trial_frame
        canvas = analysis_frame.plot_canvas.get_tk_widget()
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

    def _bound_to_mousewheel(self, event):
        self.gui.analyze_trial_frame.plot_canvas.get_tk_widget().bind_all(
                "<MouseWheel>", self._on_mousewheel)

    def _unbound_to_mousewheel(self, event):
        self.gui.analyze_trial_frame.plot_canvas.get_tk_widget().unbind_all(
                "<MouseWheel>")

    def _on_mousewheel(self, event):
        self.gui.analyze_trial_frame.plot_canvas.get_tk_widget().yview_scroll(
                -1*(event.delta), 'units')

    def update_status(self, event=None):
        new_status = self.gui.analyze_trial_frame.status_dropdown.get()
        self.trial.metadata['analysis_status'] = new_status
        self.trial.write_metadata_to_yaml(self.trial.metadata)
        self.gui.file_load_frame.file_tree.pack_forget()
        self.gui.file_load_frame.update_file_tree()
        self.gui.root.update()

    def get_analysis_progress(self, event=None):
        """Gets the analysis status of all trials and plots their status"""

        all_statuses_dict = {}
        status_values = self.trial.STATUSES
        base_dir = '/Users/adam/Documents/SenseOfTouchResearch/'
        data_location = (base_dir + 'SSN_data/*/SSN_*.nd2')
        metadata_location = (base_dir + 'SSN_ImageAnalysis/'
                             'AnalyzedData/')

        # for all subfiles ending in .nd2
        for nd2_file in glob.glob(data_location):
            experiment_id = nd2_file[-15:-4]
            metadata_file = (metadata_location +
                             experiment_id +
                             '/metadata.yaml')
            # try to load metadata
            try:
                with open(metadata_file, 'r') as yamlfile:
                    metadata = yaml.load(yamlfile)
                    # if metadata exists, get analysis status and store it
                    all_statuses_dict[experiment_id] = metadata[
                            'analysis_status']
            except FileNotFoundError:
                all_statuses_dict[experiment_id] = 'No metadata.yaml file'
            except KeyError:
                all_statuses_dict[experiment_id] = 'No analysis status yet'

        # send all analysis statuses to view module for plotting
        self.gui.plot_results_frame.plot_progress(all_statuses_dict,
                                                  status_values)

    def _display_last_test_params(self, event=None):
        analysis_frame = self.gui.analyze_trial_frame
        latest_params = self.trial.latest_test_params
        params = {**self.trial.default_test_params, **latest_params}

        analysis_frame.btm_slice_selector.delete(0, 'end')
        analysis_frame.btm_slice_selector.insert(0, params['bottom_slice'])

        analysis_frame.top_slice_selector.delete(0, 'end')
        analysis_frame.top_slice_selector.insert(0, params['top_slice'])

        analysis_frame.gaussian_blur_width.delete(0, 'end')
        analysis_frame.gaussian_blur_width.insert(0, params['gaussian_width'])

        analysis_frame.z_diameter_selector.delete(0, 'end')
        analysis_frame.z_diameter_selector.insert(
                0, params['particle_z_diameter'])

        analysis_frame.xy_diameter_selector.delete(0, 'end')
        analysis_frame.xy_diameter_selector.insert(
                0, params['particle_z_diameter'])

        analysis_frame.xy_diameter_selector.delete(0, 'end')
        analysis_frame.xy_diameter_selector.insert(
                0, params['particle_xy_diameter'])

        analysis_frame.brightness_percentile_selector.delete(0, 'end')
        analysis_frame.brightness_percentile_selector.insert(
                0, params['brightness_percentile'])

        analysis_frame.min_mass_selector.delete(0, 'end')
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
