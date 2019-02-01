#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:32:39 2019

@author: Adam Nekimken
"""
# TODO: DOCSTRINGS!!!!
import numpy as np
import tkinter as tk
# import threading
# from multiprocessing import Process, Queue
import glob
import yaml

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
        # TODO: don't instantiate model until loading trial so new trials load

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
        self.gui.analyze_trial_frame.clear_roi_btn.bind(
                "<ButtonRelease-1>", self.clear_roi)
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
        self.gui.analyze_trial_frame.status_dropdown[
                'values'] = self.trial.STATUSES
        self.gui.analyze_trial_frame.status_dropdown.bind(
                "<<ComboboxSelected>>", func=self.update_status)
        self.gui.analyze_trial_frame.plot_labels_drop.bind(
                "<<ComboboxSelected>>", func=self.update_inspection_image)
        self.gui.analyze_trial_frame.link_mitos_button.bind(
                "<ButtonRelease-1>", func=self.link_existing_particles)
        self.gui.analyze_trial_frame.calc_strain_button.bind(
                "<ButtonRelease-1>", func=self.calculate_strain)

        # Plotting frame
        self.gui.plot_results_frame.progress_plot_button.bind(
                "<ButtonRelease-1>", func=self.get_analysis_progress)
        self.gui.plot_results_frame.plot_strain_one_trial_button.bind(
                "<ButtonRelease-1>", func=self.get_analysis_progress)
        # TODO: write callback for plot strain one trial button

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
        if 'analysis_status' not in self.trial.metadata:
            self.trial.metadata['analysis_status'] = 'Not started'
            self.trial.write_metadata_to_yaml(self.trial.metadata)
        current_status = self.trial.metadata['analysis_status']
        self.gui.analyze_trial_frame.status_dropdown.set(current_status)

        # load inspection image on inspection tab and related parameters
        if load_images is True or overwrite_metadata is True:
            self.update_inspection_image()
            self.roi = [0,
                        0,
                        self.trial.image_array.shape[3],
                        self.trial.image_array.shape[2]]

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

        if self.trial.unlinked_particles_file.is_file():
            self.gui.plot_results_frame.diag_entry.insert(
                    tk.END, self.trial.mitos_from_batch.to_string())

        self._display_last_test_params()
        self.gui.notebook.select(1)

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
        time_point = int(analysis_frame.timepoint_selector.get()) - 1
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

        # TODO: get subset of array in trial file and zero out other pixels
        self.trial.run_batch(
                self.trial.image_array[:, :,
                                       self.roi[1]:self.roi[3],
                                       self.roi[0]:self.roi[2]],
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
        self.gui.plot_results_frame.diag_text.set(str(self.trial.linked_mitos))

    def run_multiple_files(self, event=None):
        fr = self.gui.file_load_frame
        overwrite_metadata = fr.overwrite_metadata_box.instate(['selected'])
        print('Running lots of files...')
        for i in range(len(self.file_list)):
            print('Loading file ', self.file_list[i][1])
            self.trial.load_trial(self.file_list[i][1],
                                  load_images=True,
                                  overwrite_metadata=overwrite_metadata)
            if 'analysis_status' not in self.trial.metadata:
                self.trial.metadata['analysis_status'] = 'Testing parameters'
                self.trial.write_metadata_to_yaml(self.trial.metadata)

            params = self.trial.latest_test_params

            print('Looking for particles...')

            self.trial.run_batch(
                images_ndarray=self.trial.image_array,
                gaussian_width=params['gaussian_width'],
                particle_z_diameter=params['particle_z_diameter'],
                particle_xy_diameter=params['particle_xy_diameter'],
                brightness_percentile=params['brightness_percentile'],
                min_particle_mass=params['min_particle_mass'],
                bottom_slice=params['bottom_slice'],
                top_slice=params['top_slice'],
                tracking_seach_radius=params['tracking_seach_radius'],
                last_timepoint=params['last_timepoint'])

            previous_status = self.trial.metadata['analysis_status']
            if (previous_status == 'No metadata.yaml file' or
                    previous_status == 'No analysis status yet' or
                    previous_status == 'Not started'):
                self.trial.metadata['analysis_status'] = 'Testing parameters'
                self.trial.write_metadata_to_yaml(self.trial.metadata)

            self.trial = ssn_trial.StrainPropagationTrial()  # clear trial var

    def run_tp_process(self):
        self.tp_process.start()
        self.tp_process.join()

    def link_existing_particles(self, event=None):
        """Link previously found particles into trajectories"""
        analysis_frame = self.gui.analyze_trial_frame
        tracking_seach_radius = int(
                analysis_frame.linking_radius_selector.get())
        last_timepoint = int(
                analysis_frame.last_time_selector.get())

        self.trial.link_mitos(tracking_seach_radius=tracking_seach_radius,
                              last_timepoint=last_timepoint)

    def calculate_strain(self, event=None):
        """Calculate strain between mitochondria as a function of time"""
        self.trial.calculate_strain()
        self.gui.plot_results_frame.plot_strain_one_trial(self.trial.strain)
        self.gui.notebook.select(2)

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
            file_load_frame.run_multiple_files_button.config(state=tk.NORMAL)
        elif len(self.file_list) == 1 and info['tags'] == ['day']:
            file_load_frame.load_trial_button.config(state=tk.DISABLED)
            file_load_frame.run_multiple_files_button.config(state=tk.NORMAL)
            # TODO: add all trials from this day to file_list
        elif len(self.file_list) == 1 and info['tags'] == ['trial']:
            file_load_frame.load_trial_button.config(state=tk.NORMAL)
            file_load_frame.run_multiple_files_button.config(state=tk.DISABLED)

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
        self.gui.plot_results_frame.diag_entry.delete(1.0, tk.END)

        # TODO: move this to view model and send data there instead?
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
                self.gui.plot_results_frame.diag_entry.insert(
                        tk.END, mitos_this_frame.to_string())
            elif self.trial.mito_candidates is not None:
                self.trial.mito_candidates.plot(
                    x='x', y='y', ax=analysis_frame.ax, color='#FB8072',
                    marker='o', linestyle='None')
                self.gui.plot_results_frame.diag_entry.insert(
                        tk.END, self.trial.mito_candidates.to_string())
        elif plot_mitos_status == 'Plot trajectories':
            if self.trial.linked_mitos is not None:
                self.gui.plot_results_frame.diag_entry.insert(
                        tk.END, self.trial.linked_mitos.to_string())
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

        analysis_frame.fig.set_size_inches(
                forward=True,
                h=image_to_display.shape[0] / self.gui.dpi,
                w=image_to_display.shape[1] / self.gui.dpi)
        analysis_frame.ax.imshow(image_to_display, interpolation='none')
        analysis_frame.ax.axis('off')
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
        # TODO: account for scroll in event location

        # create rectangle if not yet exist
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
        # TODO: convert ROI to image space or display image 1:1
        # TODO: limit ROI to actual pixel values ie no negatives

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

        # print('Selected ROI: ', self.roi)

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

    def clear_roi(self, event):
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
                    self.trial.image_array.shape[3]]

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

        analysis_frame.last_time_selector.delete(0, 'end')
        analysis_frame.last_time_selector.insert(
                0, params['last_timepoint'])

        analysis_frame.linking_radius_selector.delete(0, 'end')
        analysis_frame.linking_radius_selector.insert(
                0, params['tracking_seach_radius'])

        # TODO: save/load linking radius and last timepoint params


if __name__ == '__main__':
    controller = StrainGUIController()
    controller.run()
