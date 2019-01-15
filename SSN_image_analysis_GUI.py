#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:32:39 2019

@author: Adam Nekimken
"""
# TODO: DOCSTRINGS!!!!
# TEMP: import numpy for testing
import numpy as np
import tkinter as tk
import time
# from tkinter import ttk
# import glob
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        self.gui.file_load_frame.load_trial_button.bind("<ButtonRelease-1>",
                                                        func=self.load_trial)
        self.gui.file_load_frame.file_tree.bind(
                '<<TreeviewSelect>>', func=self.on_file_selection_changed)

        # Inspection frame
        self.gui.inspect_image_frame.update_image_btn.bind(
                "<ButtonRelease-1>", func=self.update_inspection_image)
        self.gui.inspect_image_frame.slice_selector.bind(
                "<ButtonRelease-1>", func=self.update_inspection_image)
        self.gui.inspect_image_frame.timepoint_selector.bind(
                "<ButtonRelease-1>", func=self.update_inspection_image)
        self.gui.inspect_image_frame.test_param_button.bind(
                "<ButtonRelease-1>", func=self.test_params)

    def run(self):
        self.root.title("SSN Image Analysis")
        self.root.mainloop()

    def load_trial(self, event):
        """Loads the data for this trial from disk and sends us to the
        inspection tab to start the analysis
        """
        # OPTIMIZE: load trial in a separate thread?
        # MAYBE: add popup window saying file is loading
        print('Loading file', self.file_list[0][0])
        fr = self.gui.file_load_frame
        load_images = fr.load_images_box.instate(['selected'])
        overwrite_metadata = fr.overwrite_metadata_box.instate(['selected'])
        self.trial.load_trial(self.file_list[0][0],
                              load_images=load_images,
                              overwrite_metadata=overwrite_metadata)

        # load inspection image on inspection tab and related parameters
        if load_images is True or overwrite_metadata is True:
            self.update_inspection_image()

        image_stack_size = self.trial.metadata['stack_height']
        self.gui.inspect_image_frame.slice_selector.config(
                values=list(range(1, image_stack_size + 1)))
        self.gui.inspect_image_frame.btm_slice_selector.config(
                values=list(range(1, image_stack_size + 1)))
        self.gui.inspect_image_frame.top_slice_selector.config(
                values=list(range(1, image_stack_size + 1)))
        self.gui.inspect_image_frame.timepoint_selector.config(
                values=list(range(1, image_stack_size + 1)))
        self.gui.inspect_image_frame.metadata_notes.config(
                text=self.trial.metadata['notes'])
        self.gui.inspect_image_frame.stack_height_label.config(
                text=('Stack height: ' + str(image_stack_size)))
        self.gui.inspect_image_frame.neuron_id_label.config(
                text=('Neuron: ' + self.trial.metadata['neuron']))
        self.gui.inspect_image_frame.vulva_side_label.config(
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
        # print(self.file_list)

    def update_inspection_image(self, event=None):
        insp_frame = self.gui.inspect_image_frame
        max_proj_checkbox = insp_frame.max_proj_checkbox.instate(['selected'])
        plot_labels_checkbox = insp_frame.plot_labels_box.instate(['selected'])
        selected_slice = int(insp_frame.slice_selector.get())
        selected_timepoint = int(insp_frame.timepoint_selector.get())
        inspection_ax = insp_frame.ax
        inspection_ax.clear()

        if max_proj_checkbox is True:
            # max projection
            stack = self.trial.image_array[selected_timepoint]
            image_to_display = np.amax(stack, 0)  # collapse z axis
            image_to_display = image_to_display.squeeze()
            # maxProjection = np.asarray(maxProjection)
        else:
            # single slice
            image_to_display = self.trial.image_array[
                    selected_timepoint, selected_slice]

        if plot_labels_checkbox is True and \
                self.trial.mito_candidates is not None:
            self.trial.mito_candidates.plot(
                    x='x', y='y', ax=insp_frame.ax, color='#FB8072',
                    marker='o', linestyle='None')

        insp_frame.ax.imshow(image_to_display, interpolation='none')
        insp_frame.ax.axis('off')
        insp_frame.plot_canvas.draw()

    def test_params(self, event):
        # TODO: make sure everything happens in the places that make sense
        # gather parameters
        insp_frame = self.gui.inspect_image_frame
        gaussianWidth = int(insp_frame.gaussian_blur_width.get())
        particleZDiameter = int(insp_frame.z_diameter_selector.get())
        particleXYDiameter = int(insp_frame.xy_diameter_selector.get())
        brightnessPercentile = int(
                insp_frame.brightness_percentile_selector.get())
        minParticleMass = int(insp_frame.min_particle_mass_selector.get())
        bottomSlice = int(insp_frame.btm_slice_selector.get())
        topSlice = int(insp_frame.top_slice_selector.get())
        time_point = int(insp_frame.timepoint_selector.get())
        # TODO: validate input

        # run trackpy analysis
        self.trial.mito_candidates = self.trial.test_parameters(
                gaussianWidth,
                particleZDiameter,
                particleXYDiameter,
                brightnessPercentile,
                minParticleMass,
                bottomSlice,
                topSlice,
                time_point)

        insp_frame.max_proj_checkbox.state(['selected'])
        insp_frame.plot_labels_box.state(['selected'])
        self.update_inspection_image()
        # self.mito_candidate_axes = self.trial.mito_candidates.plot(
        #         x='x', y='y', ax=insp_frame.ax, color='#FB8072', marker='o',
        #         linestyle='None')
        # insp_frame.plot_canvas.draw()

        # TODO: save results

    def _load_last_test_params(self, event=None):
        insp_frame = self.gui.inspect_image_frame
        params = self.trial.latest_test_params

        insp_frame.btm_slice_selector.delete(0, 'end')
        insp_frame.btm_slice_selector.insert(0, params['bottomSlice'])

        insp_frame.top_slice_selector.delete(0, 'end')
        insp_frame.top_slice_selector.insert(0, params['topSlice'])

        insp_frame.gaussian_blur_width.delete(0, 'end')
        insp_frame.gaussian_blur_width.insert(0, params['gaussianWidth'])

        insp_frame.z_diameter_selector.delete(0, 'end')
        insp_frame.z_diameter_selector.insert(0, params['particleZDiameter'])

        insp_frame.xy_diameter_selector.delete(0, 'end')
        insp_frame.xy_diameter_selector.insert(0, params['particleZDiameter'])

        insp_frame.xy_diameter_selector.delete(0, 'end')
        insp_frame.xy_diameter_selector.insert(0, params['particleXYDiameter'])

        insp_frame.brightness_percentile_selector.delete(0, 'end')
        insp_frame.brightness_percentile_selector.insert(
                0, params['brightnessPercentile'])

        insp_frame.min_particle_mass_selector.delete(0, 'end')
        insp_frame.min_particle_mass_selector.insert(
                0, params['minParticleMass'])


if __name__ == '__main__':
    controller = StrainGUIController()
    controller.run()
