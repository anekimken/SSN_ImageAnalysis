#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:41:44 2019

@author: adam
"""
# import numpy as np

import tkinter as tk
from tkinter import ttk
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
import warnings
import pandas as pd
# import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandastable import Table

warnings.filterwarnings("ignore",
                        message=("A value is trying to be set on a copy "
                                 "of a slice from a DataFrame"))


class SSN_analysis_GUI(tk.Frame):
    """
    This class implements a GUI for performing all parts of the image analysis
    process and plots the results. It calls other classes for each of the
    different frames in its notebook
    """

    def __init__(self, root):
        """
        Initializes the Analysis gui object
        """
        tk.Frame.__init__(self, root)
        self.root = root
        w, h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry("%dx%d+0+0" % (w, h))
        self.dpi = round(self.root.winfo_fpixels('1i'))
        self.notebook = ttk.Notebook(root)

        self.file_load_frame = FileLoadFrame(self.notebook)
        self.file_load_tab = self.notebook.add(
                self.file_load_frame, text="Load File(s)")

        # pack notebook and update so notebook has the right size
        self.notebook.pack(expand=1, fill=tk.BOTH)
        self.root.update()

        self.analyze_trial_frame = AnalyzeTrialFrame(
                self.notebook, self.dpi, self.root)
        self.notebook.add(self.analyze_trial_frame, text="Inspect Images")

        self.queue_frame = AnalysisQueueFrame(self.notebook)
        self.queue_tab = self.notebook.add(self.queue_frame,
                                           text='Analysis Queue')

        self.plot_results_frame = PlotResultsFrame(self.notebook, self.dpi,
                                                   self.root)
        self.notebook.add(self.plot_results_frame, text="Plot Results")

        self.notebook.pack(expand=1, fill=tk.BOTH)

    def create_bf_frame(self):
        self.bf_image_frame = BrightfieldImageFrame(
                self.notebook, self.dpi, self.root)
        self.notebook.insert(2, self.bf_image_frame, text="Brightfield Image")


class FileLoadFrame(tk.Frame):
    """This class implements the first tab of the GUI, which prompts the
    user to choose which file or files to analyze
    """
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent

        # define buttons for starting analysis
        self.button_frame = tk.Frame(self)

        self.overwrite_metadata_box_status = tk.IntVar(value=0)
        self.overwrite_metadata_box = ttk.Checkbutton(
                self.button_frame,
                text='Overwrite metadata?',
                variable=self.overwrite_metadata_box_status)
        self.overwrite_metadata_box.pack(side=tk.LEFT)
        self.overwrite_metadata_box.state(['!selected', '!alternate'])

        self.load_images_checkbox_status = tk.IntVar(value=0)
        self.load_images_box = ttk.Checkbutton(
                self.button_frame,
                text='Load raw images?',
                variable=self.load_images_checkbox_status)
        self.load_images_box.pack(side=tk.LEFT)
        self.load_images_box.state(['selected', '!alternate'])

        self.load_trial_button = tk.Button(self.button_frame,
                                           text="Load Trial",
                                           state=tk.DISABLED)
        self.load_trial_button.pack(side=tk.RIGHT)

        self.run_multiple_files_button = tk.Button(self.button_frame,
                                                   text="Run Multiple Files",
                                                   state=tk.DISABLED)
        self.run_multiple_files_button.pack(side=tk.RIGHT)
        self.button_frame.pack(side=tk.BOTTOM)

        self.update_file_tree()

    def update_file_tree(self):
        # TODO: Add bf image analysis status to treeg
        # now, we load all the file names into a Treeview for user selection
        data_location = '/Users/adam/Documents/SenseOfTouchResearch/SSN_data/*'
        queue_location = ('/Users/adam/Documents/SenseOfTouchResearch/'
                          'SSN_ImageAnalysis/')
        the_queue = queue_location + 'analysis_queue.yaml'
        queue_result_location = queue_location + 'review_queue.yaml'
        experiment_days = glob.iglob(data_location)

        self.file_tree = tk.ttk.Treeview(self, height=37,
                                         columns=("status", "rating", "queue"))
        self.file_tree.heading("status", text="Analysis Status")
        self.file_tree.heading("queue", text="Queue status")
        self.file_tree.heading("rating",
                               text="Rating to prioritize analysis")

        with open(the_queue, 'r') as queue_file:
            entire_queue = yaml.load_all(queue_file)
            self.trials_in_queue = []
            for queue_member in entire_queue:
                self.trials_in_queue.append(queue_member['experiment_id'])

        with open(queue_result_location, 'r') as queue_result:
            all_queue_results = yaml.load_all(queue_result)
            self.trials_for_review = []
            for trial in all_queue_results:
                self.trials_for_review.append(trial['experiment_id'])

        for day in experiment_days:
            try:
                int(day[-8:])  # check to make sure this dir is an experiment
                day_item = self.file_tree.insert('', 'end', text=day[-8:],
                                                 tags='day')
                trials = glob.iglob(day + '/*.nd2')
                trials = [trial for trial in trials if '_bf' not in trial]
                for trial in trials:
                    trial_parts = trial.split('/')
                    iid = self.file_tree.insert(day_item, 'end',
                                                text=trial_parts[-1])
                    metadata_file_path = ('/Users/adam/Documents/'
                                          'SenseOfTouchResearch/'
                                          'SSN_ImageAnalysis/AnalyzedData/' +
                                          trial[-15:-4] + '/metadata.yaml')
                    try:
                        metadata = self.load_metadata_from_yaml(
                                metadata_file_path)
                        if 'analysis_status' in metadata:
                            analysis_status = metadata['analysis_status']
                        else:
                            analysis_status = 'no parameters tested'
                        if 'trial_rating' in metadata:
                            rating = metadata['trial_rating']
                        else:
                            rating = None
                        if metadata['Experiment_id'] in self.trials_in_queue:
                            queue_status = 'In queue'
                        elif metadata[
                                'Experiment_id'] in self.trials_for_review:
                            queue_status = 'Ready for review'
                        else:
                            queue_status = ''
                    except FileNotFoundError:
                        analysis_status = 'no metadata.yaml file'
                    if analysis_status[:6] == 'Failed':
                        color_tag = 'failed'
                    elif analysis_status == 'Strain calculated':
                        color_tag = 'done'
                    else:
                        color_tag = 'working'
                    self.file_tree.item(
                            iid,
                            values=(analysis_status, rating, queue_status),
                            tags=(trial, color_tag))

            except ValueError:  # we get here if dir name is not a number
                pass  # and we ignore these dirs
        self.file_tree.tag_configure('failed', background='indian red')
        self.file_tree.tag_configure('working', background='pale goldenrod')
        self.file_tree.tag_configure('done', background='pale green')
        self.file_tree.pack(fill=tk.BOTH, anchor=tk.N)

    def load_metadata_from_yaml(self, metadata_file_path: str) -> dict:
        """Loads metadata from an existing yaml file."""
        with open(metadata_file_path, 'r') as yamlfile:
            metadata = yaml.load(yamlfile)

        return metadata


class AnalyzeImageFrame(tk.Frame):
    def __init__(self, parent, screen_dpi, root):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.root = root

        # Variables for drawing ROI
        self.rect = None
        self.drag_btn_size = 5  # radius of circle for dragging ROI corner
        self.roi_corners = [None, None, None, None]
        self.roi = [None, None, None, None]

        # get size for making figure
        notebook_height = self.parent.winfo_height() - 100
        self.notebook_height_in = notebook_height / screen_dpi

        # Create a figure and a canvas for showing images
        self.fig = mpl.figure.Figure(figsize=(600 / screen_dpi,
                                              1200 / screen_dpi))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.plot_canvas = FigureCanvasTkAgg(self.fig, self)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().grid(sticky=tk.W + tk.N,
                                              row=0,
                                              column=0,
                                              rowspan=3)

        self.scrollbar = tk.Scrollbar(
                self, command=self.plot_canvas.get_tk_widget().yview)
        self.scrollbar.grid(row=0, column=1, sticky=tk.NE + tk.SE)

        self.plot_canvas.get_tk_widget().config(
                yscrollcommand=self.scrollbar.set,
                yscrollincrement=5)

        # Creat a notebook to put all the controls and parameters in
        self.analysis_notebook = ttk.Notebook(self)
        self.param_frame = tk.Frame(self.analysis_notebook)
        self.param_frame_tab = self.analysis_notebook.add(
                self.param_frame, text="Adjust Parameters")
        self.analysis_notebook.grid(row=0, column=2, sticky=tk.N)
        self.root.update()


class AnalyzeTrialFrame(AnalyzeImageFrame):
    def __init__(self, parent, screen_dpi, root):
        AnalyzeImageFrame.__init__(self, parent, screen_dpi, root)

        param_frame_row = 0

        # Clear ROI button
        self.clear_roi_btn = ttk.Button(self.param_frame,
                                        text='Clear ROI')
        self.clear_roi_btn.grid(row=param_frame_row, column=1)

        # Optional max projection
        self.max_proj_checkbox_status = tk.IntVar(value=1)
        self.max_proj_checkbox = ttk.Checkbutton(
                self.param_frame,
                text='Max projection',
                variable=self.max_proj_checkbox_status)
        self.max_proj_checkbox.grid(
                row=param_frame_row, column=2)
        self.max_proj_checkbox.state(['selected', '!alternate'])

        param_frame_row += 1

        # Optional particle labels
        self.part_label_checkbox_status = tk.IntVar(value=1)
        self.part_label_checkbox = ttk.Checkbutton(
                self.param_frame,
                text='Label particles',
                variable=self.part_label_checkbox_status)
        self.part_label_checkbox.grid(
                row=param_frame_row, column=2)
        self.part_label_checkbox.state(['!selected', '!alternate'])

        param_frame_row += 1

        # Select slice to show
        self.slice_selector_label = ttk.Label(self.param_frame,
                                              text='Choose slice to display: ')
        self.slice_selector_label.grid(row=param_frame_row, column=1)
        self.slice_selector = tk.Spinbox(self.param_frame,
                                         values=(1, 2), width=5)
        self.slice_selector.grid(row=param_frame_row, column=2)
        param_frame_row += 1

        # Select timepoint to show
        self.timepoint_selector_label = ttk.Label(
                self.param_frame, text='Choose timepoint to display: ')
        self.timepoint_selector_label.grid(row=param_frame_row, column=1)
        self.timepoint_selector = tk.Spinbox(self.param_frame,
                                             values=(1, 2),  width=5)
        self.timepoint_selector.grid(row=param_frame_row, column=2)
        param_frame_row += 1

        # dropdown for toggling mito candidate labels image
        self.plot_labels_drop = ttk.Combobox(
                self.param_frame,
                state="readonly",
                values=[
                        'Plot trajectories',
                        'Linked mitos for this stack',
                        'Unlinked mitos for this stack',
                        'Mitos from param test',
                        'No overlay'])
        self.plot_labels_drop.grid(column=1, row=param_frame_row)

        # Button for updating image
        self.update_image_btn = ttk.Button(self.param_frame,
                                           text='Update image')
        self.update_image_btn.grid(row=param_frame_row, column=2, sticky=tk.N)
        param_frame_row += 1

        # Label for section with parameters
        self.enter_params_label = ttk.Label(
                self.param_frame, text='Parameters for locating particles: ')
        self.enter_params_label.grid(
                row=param_frame_row, column=1, columnspan=2)
        param_frame_row += 1

        # Selector for bottom slice
        self.btm_slice_label = ttk.Label(self.param_frame,
                                         text='Bottom slice: ')
        self.btm_slice_label.grid(row=param_frame_row, column=1)
        self.btm_slice_selector = tk.Spinbox(self.param_frame,
                                             values=(1, 2), width=5)
        self.btm_slice_selector.grid(row=param_frame_row, column=2)
        param_frame_row += 1

        # Selector for top slice
        self.top_slice_label = ttk.Label(self.param_frame,
                                         text='Top slice: ')
        self.top_slice_label.grid(row=param_frame_row, column=1)
        self.top_slice_selector = tk.Spinbox(self.param_frame,
                                             values=(1, 2), width=5)
        self.top_slice_selector.grid(row=param_frame_row, column=2)
        param_frame_row += 1

        # Selector for last timepoint
        self.last_time_label = ttk.Label(self.param_frame,
                                         text='Final timepoint to analyze: ')
        self.last_time_label.grid(row=param_frame_row, column=1)
        self.last_time_selector = tk.Spinbox(self.param_frame,
                                             values=(1, 2), width=5)
        self.last_time_selector.grid(row=param_frame_row, column=2)
        param_frame_row += 1

        # gaussian_width,
        self.gaussian_blur_width_label = ttk.Label(
                self.param_frame, text='Width of Gaussian blur kernel: ')
        self.gaussian_blur_width_label.grid(row=param_frame_row, column=1)
        self.gaussian_blur_width = tk.Spinbox(self.param_frame,
                                              values=list(range(0, 10)),
                                              width=5)
        self.gaussian_blur_width.grid(row=param_frame_row, column=2)
        param_frame_row += 1

        # particle_z_diameter,
        self.z_diameter_label = ttk.Label(self.param_frame,
                                          text='Particle z diameter: ')
        self.z_diameter_label.grid(row=param_frame_row, column=1)
        self.z_diameter_selector = tk.Spinbox(self.param_frame,
                                              values=list(range(1, 45, 2)),
                                              width=5)
        self.z_diameter_selector.grid(row=param_frame_row, column=2)
        param_frame_row += 1

        #                 particle_xy_diameter,
        self.xy_diameter_label = ttk.Label(self.param_frame,
                                           text='Particle xy diameter: ')
        self.xy_diameter_label.grid(row=param_frame_row, column=1)
        self.xy_diameter_selector = tk.Spinbox(self.param_frame,
                                               values=list(range(1, 45, 2)),
                                               width=5)
        self.xy_diameter_selector.grid(row=param_frame_row, column=2)
        param_frame_row += 1

        #                 brightness_percentile,
        self.brightness_percentile_label = ttk.Label(
                self.param_frame, text='Brightness percentile: ')
        self.brightness_percentile_label.grid(row=param_frame_row, column=1)
        self.brightness_percentile_selector = tk.Spinbox(
                self.param_frame, values=list(range(1, 100)), width=5)
        self.brightness_percentile_selector.grid(row=param_frame_row, column=2)
        param_frame_row += 1

        #                 min_particle_mass,
        self.min_particle_mass_label = ttk.Label(
                self.param_frame, text='Minimum particle mass: ')
        self.min_particle_mass_label.grid(row=param_frame_row, column=1)
        self.min_mass_selector = tk.Entry(self.param_frame, width=5)
        self.min_mass_selector.grid(row=param_frame_row, column=2)
        param_frame_row += 1

        # radius for linking particles
        self.linking_radius_label = ttk.Label(
                self.param_frame, text='Linking search radius: ')
        self.linking_radius_label.grid(row=param_frame_row, column=1)
        self.linking_radius_selector = tk.Spinbox(
                self.param_frame, values=list(range(1, 100)), width=5)
        self.linking_radius_selector.grid(row=param_frame_row, column=2)
        param_frame_row += 1

        # Button to move to test parameters
        self.test_param_button = ttk.Button(
                self.param_frame, text='Test params')
        self.test_param_button.grid(
                row=param_frame_row, column=1)

        # Button to analyze all timepoints with these parameters
        self.full_analysis_button = ttk.Button(
                self.param_frame, text='Run full analysis')
        self.full_analysis_button.grid(
                row=param_frame_row, column=2)
        param_frame_row += 1

        # Button to link previously found mitochondria
        self.link_mitos_button = ttk.Button(
                self.param_frame, text='Link existing data')
        self.link_mitos_button.grid(
                row=param_frame_row, column=1)

        # Button to add trial to queue for later analysis
        self.add_to_queue_btn = ttk.Button(
                self.param_frame, text='Add to queue')
        self.add_to_queue_btn.grid(row=param_frame_row, column=2)
        param_frame_row += 1

        # Button to remove trial from queue results
        self.remove_q_result = ttk.Button(
                self.param_frame, text='Done reviewing', state=tk.DISABLED)
        self.remove_q_result.grid(row=param_frame_row, column=2)

        # Button to calculate strain
        self.calc_strain_button = ttk.Button(
                self.param_frame, text='Calculate strain')
        self.calc_strain_button.grid(row=param_frame_row, column=1,
                                     columnspan=1)
        param_frame_row += 1

        # Controls for saving data etc.
        save_frame_row = 0

        # Show notes from metadata
        self.metadata_notes_label = ttk.Label(self.param_frame,
                                              text='Notes from metadata:')
        self.metadata_notes_label.grid(row=save_frame_row, column=3)
        save_frame_row += 1
        self.metadata_notes = tk.Message(self.param_frame)
        self.metadata_notes.grid(row=save_frame_row, column=3, rowspan=5)
        save_frame_row += 5

        # Show height of stack
        self.stack_height_label = ttk.Label(self.param_frame,
                                            text='Stack height:')
        self.stack_height_label.grid(row=save_frame_row, column=3)
        save_frame_row += 1

        # Show which neuron is being tested
        self.neuron_id_label = ttk.Label(self.param_frame,
                                         text='Neuron:')
        self.neuron_id_label.grid(row=save_frame_row, column=3)
        save_frame_row += 1

        # Show which side the worm's vulva is on
        self.vulva_side_label = ttk.Label(self.param_frame,
                                          text='Vulva side:')
        self.vulva_side_label.grid(row=save_frame_row, column=3)
        save_frame_row += 1

        # Update status
        self.status_dropdown = ttk.Combobox(
                self.param_frame,
                state="readonly",
                values=['No options loaded yet'])
        self.status_dropdown.grid(column=3, row=save_frame_row)
        save_frame_row += 1

        self.notes_entry = ttk.Entry(master=self.param_frame)
        self.note_entry_default = 'Notes for analysis run'
        self.notes_entry.insert(0, self.note_entry_default)

        def click_note_entry(event=None):
            if event.widget.get() == self.note_entry_default:
                event.widget.delete(0, tk.END)

        self.notes_entry.bind("<ButtonRelease-1>", click_note_entry)
        self.notes_entry.bind("<FocusIn>", click_note_entry)
        self.notes_entry.grid(row=save_frame_row, column=3, sticky=tk.N+tk.S)
        save_frame_row += 1

        self.explore_data_frame = tk.Frame(self.analysis_notebook)
        self.df_tab = self.analysis_notebook.add(
                self.explore_data_frame, text="Explore Data")
        self.dataframe = None  # TableModel.getSampleData()
        self.dataframe_widget = Table(self.explore_data_frame,
                                      dataframe=self.dataframe)
        self.dataframe_widget.show()

        self.param_history_frame = tk.Frame(self.analysis_notebook)
        self.param_history_tab = self.analysis_notebook.add(
                self.param_history_frame, text="Analysis History")
        self.param_history = tk.Text(self.param_history_frame,
                                     state=tk.DISABLED)
        self.param_history.grid(row=0, column=0, sticky=tk.N + tk.S)
        self.history_scroll = tk.Scrollbar(self.param_history_frame)
        self.history_scroll.grid(row=0, column=1, sticky=tk.N + tk.S)
        self.history_scroll.config(command=self.param_history.yview)
        self.param_history.config(yscrollcommand=self.history_scroll.set)

        self.histogram_frame = tk.Frame(self.analysis_notebook)
        self.histogram_tab = self.analysis_notebook.add(
                self.histogram_frame, text="Histogram")
        self.histogram = mpl.figure.Figure()
        self.hist_ax = self.histogram.add_axes([0.1, 0.1, 0.8, 0.8])
        self.histogram_canvas = FigureCanvasTkAgg(self.histogram,
                                                  master=self.histogram_frame)

        self.histogram_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.histogram_canvas.draw()

        self.min_pixel_disp_label = ttk.Label(
                self.histogram_frame, text='Minimum')
        self.min_pixel_disp_label.pack(side=tk.LEFT)
        self.min_pixel_disp = tk.Spinbox(self.histogram_frame,
                                         values=list(range(0, 299)))
        self.min_pixel_disp.pack(side=tk.LEFT)
        self.max_pixel_disp = tk.Spinbox(self.histogram_frame,
                                         values=list(range(1, 300)))
        self.max_pixel_disp.insert(tk.END, 300)
        self.max_pixel_disp.pack(side=tk.LEFT)
        self.max_pixel_disp_label = ttk.Label(
                self.histogram_frame, text='Maximum')
        self.max_pixel_disp_label.pack(side=tk.LEFT)


class BrightfieldImageFrame(AnalyzeImageFrame):
    def __init__(self, parent, screen_dpi, root):
        AnalyzeImageFrame.__init__(self,  parent, screen_dpi, root)

        info_frame_row = 0

        # Show notes from metadata
        self.vulva_orientation = ttk.Label(
                self.param_frame,
                text='Vulva oriented')
        self.vulva_orientation.grid(row=info_frame_row, column=3)
        info_frame_row += 1

        self.neuron_label = ttk.Label(
                self.param_frame,
                text='Neuron')
        self.neuron_label.grid(row=info_frame_row, column=3)
        info_frame_row += 2

        self.actuator_location_label = ttk.Label(
                self.param_frame,
                text='I do not know yet')
        self.actuator_location_label.grid(row=info_frame_row, column=3)
        info_frame_row += 1

        # TODO: clear actuator location button
        # TODO: button for saving actuator location


class AnalysisQueueFrame(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.queue_location = ('/Users/adam/Documents/SenseOfTouchResearch/'
                               'SSN_ImageAnalysis/')
        self.queue_tree = tk.ttk.Treeview(self, height=37,
                                          columns=("param_val", "rating"))
        self.queue_tree.heading("param_val", text="Value")
        self.queue_tree.heading("rating", text="Experimenter's rating")

        # define buttons for starting analysis
        self.button_frame = tk.Frame(self)

        self.run_queue_button = tk.Button(self.button_frame,
                                          text="Run Queue",
                                          state=tk.DISABLED)
        self.run_queue_button.pack(side=tk.RIGHT)
        self.button_frame.pack(side=tk.BOTTOM)

        def update_queue(event=None):
            the_queue = self.queue_location + 'analysis_queue.yaml'
            with open(the_queue, 'r') as queue_file:
                entire_queue = yaml.load_all(queue_file)
                for queue_member in entire_queue:
                    self.run_queue_button.config(state=tk.NORMAL)
                    experiment_id = queue_member['experiment_id']

                    # add to tree, add all but experiment ID to lower level
                    queue_item = self.queue_tree.insert(
                            '', 'end',
                            text=experiment_id)
                    metadata_file_path = ('/Users/adam/Documents/'
                                          'SenseOfTouchResearch/'
                                          'SSN_ImageAnalysis/AnalyzedData/' +
                                          experiment_id + '/metadata.yaml')
                    try:
                        with open(metadata_file_path, 'r') as yamlfile:
                            metadata = yaml.load(yamlfile)
                        if 'trial_rating' in metadata:
                            rating = metadata['trial_rating']
                        else:
                            rating = None
                    except FileNotFoundError:
                            pass
                    self.queue_tree.item(queue_item, values=('', rating))

                    for param, value in queue_member.items():
                        if param == 'experiment_id':
                            # we already have this in the tree
                            pass
                        elif param == 'roi':
                            roi_str = (str(value[0]) + ', ' +
                                       str(value[1]) + ', ' +
                                       str(value[2]) + ', ' +
                                       str(value[3]))
                            self.queue_tree.insert(queue_item, 'end',
                                                   text=param,
                                                   values=(roi_str, ''))
                        else:
                            self.queue_tree.insert(queue_item, 'end',
                                                   text=param,
                                                   values=(value, ''))

            self.queue_tree.pack(fill=tk.BOTH, anchor=tk.N)
        self.bind("<Visibility>", update_queue)


class PlotResultsFrame(AnalyzeImageFrame):
    def __init__(self, parent, screen_dpi, root):
        AnalyzeImageFrame.__init__(self, parent, screen_dpi, root)

        # Reset fig size since we don't need a big figure here
        self.plot_canvas.get_tk_widget().config(height=600)
        self.ax.set_position([0.1, 0.2, 0.8, 0.7])

        self.progress_plot_button = ttk.Button(self.param_frame,
                                               text='Plot analysis progress')
        self.progress_plot_button.grid(row=0,  column=1)

        # Tab for strain plot for one trial
        self.strain_plot_frame = tk.Frame(master=self.analysis_notebook)
        self.strain_one_trial_tab = self.analysis_notebook.add(
                self.strain_plot_frame, text='Plot strain')
        self.plot_strain_one_trial_button = ttk.Button(
                self.strain_plot_frame, text='Strain from one trial')
        self.plot_strain_one_trial_button.grid(row=1, column=0)
        self.plot_strain_by_actuation_btn = ttk.Button(
                self.strain_plot_frame, text='Plot strain by actuation')
        self.plot_strain_by_actuation_btn.grid(row=1, column=1)
        self.plot_xz_disp_btn = ttk.Button(
                self.strain_plot_frame, text='Plot xz displacement')
        self.plot_xz_disp_btn.grid(row=2, column=1)

        self.update_plot_strain_tree()

        # TODO: add buttons for interesting plots

    def update_plot_strain_tree(self):
        # Load trials with strain calculated into file tree
        data_location = ('/Users/adam/Documents/'
                         'SenseOfTouchResearch/SSN_data/*')
        experiment_days = glob.iglob(data_location)

        self.plot_strain_tree = tk.ttk.Treeview(self.strain_plot_frame,
                                                height=20, columns=("status"))
        self.plot_strain_tree.heading("status", text="Analysis Status")

        for day in experiment_days:
            try:
                int(day[-8:])  # check to make sure this dir has trials
                trials = glob.iglob(day + '/*.nd2')
                for trial in trials:
                    trial_parts = trial.split('/')
                    iid = self.plot_strain_tree.insert(
                                    '', 'end',
                                    text=trial_parts[-1],
                                    tags=trial)
                    metadata_path = ('/Users/adam/Documents/'
                                     'SenseOfTouchResearch/'
                                     'SSN_ImageAnalysis/AnalyzedData/' +
                                     trial[-15:-4] + '/metadata.yaml')
                    try:
                        metadata = self.load_metadata_from_yaml(
                                metadata_path)
                        if metadata['analysis_status'] != 'Strain calculated':
                            self.plot_strain_tree.delete(iid)
                        else:
                            self.plot_strain_tree.item(
                                    iid,
                                    values=(metadata['analysis_status'],))
                    except FileNotFoundError:
                        self.plot_strain_tree.delete(iid)

            except ValueError:  # we get here if dir name is not a number
                pass  # and we ignore these dirs
        self.plot_strain_tree.grid(row=0, column=0, columnspan=2)

    def plot_progress(self, all_statuses_dict: dict, status_values: list):
        statuses_to_ignore = ['No metadata.yaml file',
                              'No analysis status yet',
                              'Not started', 'Testing parameters for batch']
        for key in statuses_to_ignore:
            status_values.remove(key)
        alm_counts = dict.fromkeys(status_values, 0)
        avm_counts = dict.fromkeys(status_values, 0)
        pvm_counts = dict.fromkeys(status_values, 0)

        for experiment_id, status in all_statuses_dict.items():
            if status[1] == 'ALM':
                if status[0] not in statuses_to_ignore:
                    alm_counts[status[0]] += 1
            elif status[1] == 'AVM':
                if status[0] not in statuses_to_ignore:
                    avm_counts[status[0]] += 1
            elif status[1] == 'PVM':
                if status[0] not in statuses_to_ignore:
                    pvm_counts[status[0]] += 1

        alm_vals = np.fromiter(alm_counts.values(), dtype=int)
        avm_vals = np.fromiter(avm_counts.values(), dtype=int)
        pvm_vals = np.fromiter(pvm_counts.values(), dtype=int)

        self.ax.bar(range(len(alm_counts)), alm_vals,
                    align='center')
        self.ax.bar(range(len(avm_counts)), avm_vals,
                    align='center', bottom=alm_vals)
        self.ax.bar(range(len(pvm_counts)), pvm_vals,
                    align='center', bottom=alm_vals + avm_vals)
        self.ax.set_xticks(range(len(alm_counts)))
        self.ax.set_xticklabels(list(alm_counts.keys()),
                                rotation=30, horizontalalignment='right')
        self.ax.legend(['ALM', 'AVM', 'PVM'])

        self.plot_canvas.draw()

    def plot_strain_one_trial(self, strain, ycoords):
        """Plots the strain results from a np.ndarray"""

        self.ax.clear()
#        self.ax.plot(ycoords, strain)
        for i in range(len(strain)):
            self.ax.step(ycoords[i], strain[i])

        self.plot_canvas.draw()

    def plot_strain_by_actuation(self, strain, ycoords, pressure):
        """Averages strain by actuation pressure and plots result"""

        self.ax.clear()
        index = 0
        temp_dict = []
        for i in range(len(strain)):
            for j in range(len(strain[0])):
                temp_dict.append({'strain': strain[i][j],
                                  'ycoords': ycoords[i][j],
                                  'pressure': pressure[i],
                                  'stack_num': i})
                index += 1
        df = pd.DataFrame(temp_dict)

        df.loc[df['pressure'] == 0].groupby(['stack_num']).plot(
                x='ycoords', y='strain',
                ax=self.ax, color='red', drawstyle="steps")
        df.loc[df['pressure'] == 300].groupby(['stack_num']).plot(
                x='ycoords', y='strain',
                ax=self.ax, color='green', drawstyle="steps")

        self.plot_canvas.draw()

    def plot_xz_displacements(self, xz_displacement, ycoords, pressure):
        self.ax.clear()
        index = 0
        temp_dict = []
        for i in range(len(xz_displacement)):
            for j in range(len(xz_displacement[0])):
                temp_dict.append({'xz_displacement': xz_displacement[i][j],
                                  'ycoords': ycoords[i][j],
                                  'pressure': pressure[i],
                                  'stack_num': i})
                index += 1
        df = pd.DataFrame(temp_dict)

        df.loc[df['pressure'] == 0].groupby(['stack_num']).plot(
                x='ycoords', y='xz_displacement',
                ax=self.ax, color='red')
        df.loc[df['pressure'] == 300].groupby(['stack_num']).plot(
                x='ycoords', y='xz_displacement',
                ax=self.ax, color='green')

        self.plot_canvas.draw()

    def load_metadata_from_yaml(self, metadata_file_path: str) -> dict:
        """Loads metadata from an existing yaml file."""
        with open(metadata_file_path, 'r') as yamlfile:
            metadata = yaml.load(yamlfile)

        return metadata
    # TODO: controls for selecting what data are plotted


if __name__ == '__main__':
    import ssn_image_analysis_gui_controller
    controller = ssn_image_analysis_gui_controller.StrainGUIController()
    controller.run()
