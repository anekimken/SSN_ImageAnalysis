#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:32:39 2019

@author: Adam Nekimken
"""
# TODO: DOCSTRINGS!!!!
import numpy as np
import tkinter as tk
import threading
from multiprocessing import Process, Queue

import strain_propagation_trial as ssn_trial
import strain_propagation_view as ssn_view


class StrainGUIController:
    """Controller for running strain propagation analysis program.

    Uses Model-View-Controller architecture for doing analysis of data for
    the strain propagation project.
    """
    def __init__(self):
        self.root = tk.Tk()

        # Instantiate Model
        self.trial = ssn_trial.StrainPropagationTrial()

        # Instantiate View
        self.gui = ssn_view.SSN_analysis_GUI(self.root)

        # Bind UI elements to functions
        # Load trial tab
        self.gui.file_load_frame.load_trial_button.bind(
                "<ButtonRelease-1>", func=self.load_trial)
        self.gui.file_load_frame.file_tree.bind(
                '<<TreeviewSelect>>', func=self.on_file_selection_changed)

        # Analysis frame
        self.gui.analyze_trial_frame.update_image_btn.bind(
                "<ButtonRelease-1>", func=self.update_inspection_image)
        self.gui.analyze_trial_frame.slice_selector.bind(
                "<ButtonRelease-1>", func=self.spinbox_delay_then_update_image)
        self.gui.analyze_trial_frame.timepoint_selector.bind(
                "<ButtonRelease-1>", func=self.spinbox_delay_then_update_image)
        self.gui.analyze_trial_frame.test_param_button.bind(
                "<ButtonRelease-1>", func=self.test_params)
        self.gui.analyze_trial_frame.full_analysis_button.bind(
                "<ButtonRelease-1>", func=self.run_full_analysis)
        self.gui.analyze_trial_frame.status_dropdown.bind(
                "<<ComboboxSelected>>", func=self.update_status)
        self.gui.analyze_trial_frame.plot_labels_drop.bind(
                "<<ComboboxSelected>>", func=self.update_inspection_image)

    def run(self):
        self.root.title("SSN Image Analysis")
        self.root.mainloop()

    # def start_load_trial(self, event=None):
    #     load_thread = threading.Thread(target=self.load_trial)
    #     load_thread.start()

    def load_trial(self, event=None):
        """Loads the data for this trial from disk and sends us to the
        inspection tab to start the analysis
        """
        # OPTIMIZE: load trial in a separate thread?
        # MAYBE: add popup window saying file is loading
        print('Loading file', self.file_list[0][1])
        fr = self.gui.file_load_frame
        load_images = fr.load_images_box.instate(['selected'])
        overwrite_metadata = fr.overwrite_metadata_box.instate(['selected'])

        # load_thread = threading.Thread(
        #         name='load_file_thread',
        #         target=self.trial.load_trial,
        #         args=(self.file_list[0][0]),
        #         kwargs={'load_images': load_images,
        #                 'overwrite_metadata': overwrite_metadata})
        # load_thread.start()

        self.trial.load_trial(self.file_list[0][1],
                              load_images=load_images,
                              overwrite_metadata=overwrite_metadata)
        current_status = self.trial.metadata['analysis_status']
        self.gui.analyze_trial_frame.status_dropdown.set(current_status)
        # print(self.trial.metadata['analysis_status'])

        # load inspection image on inspection tab and related parameters
        if load_images is True or overwrite_metadata is True:
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

        self._load_last_test_params()
        self.gui.notebook.select(1)

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
            self.file_list.append(info['values'])
        if len(self.file_list) > 1:
            file_load_frame.load_trial_button.config(state=tk.DISABLED)
            file_load_frame.run_batch_button.config(state=tk.NORMAL)
        elif len(self.file_list) == 1 and info['tags'] == ['day']:
            file_load_frame.load_trial_button.config(state=tk.DISABLED)
            file_load_frame.run_batch_button.config(state=tk.NORMAL)
            # TODO: add all trials from this day to file_list
        elif len(self.file_list) == 1 and info['tags'] == ['trial']:
            file_load_frame.load_trial_button.config(state=tk.NORMAL)
            file_load_frame.run_batch_button.config(state=tk.DISABLED)

    def spinbox_delay_then_update_image(self, event=None):
        self.gui.analyze_trial_frame.after(1, self.update_inspection_image)

    def update_inspection_image(self, event=None):
        analysis_frame = self.gui.analyze_trial_frame
        plot_mitos_status = analysis_frame.plot_labels_drop.get()
        max_proj_checkbox = analysis_frame.max_proj_checkbox.instate(
                ['selected'])
        selected_slice = int(analysis_frame.slice_selector.get())
        selected_timepoint = int(analysis_frame.timepoint_selector.get()) - 1
        inspection_ax = analysis_frame.ax
        inspection_ax.clear()

        if max_proj_checkbox is True:  # max projection
            stack = self.trial.image_array[selected_timepoint]
            image_to_display = np.amax(stack, 0)  # collapse z axis
            image_to_display = image_to_display.squeeze()
        else:  # single slice
            image_to_display = self.trial.image_array[
                    selected_timepoint, selected_slice]

        if plot_mitos_status == 'Plot mitochondria for this stack':
            if (self.trial.linked_mitos is not None and
                    selected_timepoint - 1 <=
                    max(self.trial.linked_mitos['frame'])):
                mitos_this_frame = self.trial.mitos_from_batch.loc[
                        self.trial.mitos_from_batch[
                                'frame'] == selected_timepoint]
                mitos_this_frame.plot(
                            x='x', y='y', ax=analysis_frame.ax,
                            color='#FB8072', marker='o', linestyle='None')
            elif self.trial.mito_candidates is not None:
                self.trial.mito_candidates.plot(
                    x='x', y='y', ax=analysis_frame.ax, color='#FB8072',
                    marker='o', linestyle='None')
        elif plot_mitos_status == 'Plot trajectories':
            if self.trial.linked_mitos is not None:
                try:
                    theCount = 0  # ah ah ah ah
                    for i in range(max(self.trial.linked_mitos['particle'])):
                        thisParticleTraj = self.trial.linked_mitos.loc[
                                self.trial.linked_mitos['particle'] == i+1]
                        if not thisParticleTraj.empty:
                            theCount += 1
                            thisParticleTraj.plot(
                                    x='x',
                                    y='y',
                                    ax=analysis_frame.ax,
                                    color='#FB8072')
                            analysis_frame.ax.text(
                                    thisParticleTraj['x'].mean() + 15,
                                    thisParticleTraj['y'].mean(),
                                    str(int(thisParticleTraj.iloc[0][
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

        analysis_frame.ax.imshow(image_to_display, interpolation='none')
        analysis_frame.ax.axis('off')
        analysis_frame.plot_canvas.draw()

    # def test_params(self, event):
    #     # TODO: decide if I really want threads, and, if so, organize them
    #     thread = threading.Thread(target=self.test_params_separate_thread)
    #     thread.start()

    def test_params(self, event=None):
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
        time_point = int(analysis_frame.timepoint_selector.get())
        # TODO: validate input

        # run trackpy analysis in new thread
        # tp_thread = threading.Thread(name='trackpy_locate_thread',
        #                              target=self.trial.test_parameters,
        #                              args=(gaussian_width,
        #                                    particle_z_diameter,
        #                                    particle_xy_diameter,
        #                                    brightness_percentile,
        #                                    min_particle_mass,
        #                                    bottom_slice,
        #                                    top_slice,
        #                                    time_point))
        # tp_thread.start()

        self.trial.mito_candidates = self.trial.test_parameters(
                gaussian_width,
                particle_z_diameter,
                particle_xy_diameter,
                brightness_percentile,
                min_particle_mass,
                bottom_slice,
                top_slice,
                time_point)

        analysis_frame.max_proj_checkbox.state(['selected'])
        analysis_frame.plot_labels_drop.set('Plot mitochondria for this stack')
        self.update_inspection_image()

        # TODO: save results

    def run_full_analysis(self, event=None):
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

        # TODO: run as producer in separate process
        # TODO: Create consumer in separate thread that waits for this to stop
        # https://stackoverflow.com/questions/25204579/python-multiprocessing-and-gui
        # TODO: kill button for process running analysis
        self.trial.run_batch(
                gaussian_width,
                particle_z_diameter,
                particle_xy_diameter,
                brightness_percentile,
                min_particle_mass,
                bottom_slice,
                top_slice,
                tracking_seach_radius,
                last_timepoint)

        analysis_frame.max_proj_checkbox.state(['selected'])
        analysis_frame.plot_labels_drop.set('Plot trajectories')
        self.update_inspection_image()

    def update_status(self, event=None):
        new_status = self.gui.analyze_trial_frame.status_dropdown.get()
        self.trial.metadata['analysis_status'] = new_status
        self.trial.write_metadata_to_yaml(self.trial.metadata)
        self.gui.file_load_frame.file_tree.pack_forget()
        self.gui.file_load_frame.update_file_tree()
        self.gui.root.update()

    def _load_last_test_params(self, event=None):
        analysis_frame = self.gui.analyze_trial_frame
        params = self.trial.latest_test_params

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

        # TODO: save/load linking radius parameter


if __name__ == '__main__':
    controller = StrainGUIController()
    controller.run()
