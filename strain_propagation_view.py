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
import yaml
# import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandastable import Table


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

        self.plot_results_frame = PlotResultsFrame(self.notebook, self.dpi)
        self.notebook.add(self.plot_results_frame, text="Plot Results")

        self.notebook.pack(expand=1, fill=tk.BOTH)


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
                text='Load images?',
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
        # now, we load all the file names into a Treeview for user selection
        data_location = '/Users/adam/Documents/SenseOfTouchResearch/SSN_data/*'
        experiment_days = glob.iglob(data_location)

        self.file_tree = tk.ttk.Treeview(self, height=37,
                                         columns=("status", "rating"))
        self.file_tree.heading("status", text="Analysis Status")
        self.file_tree.heading("rating",
                               text="Rating to prioritize analysis")
        for day in experiment_days:
            try:
                int(day[-8:])  # check to make sure this dir is an experiment
                day_item = self.file_tree.insert('', 'end', text=day[-8:],
                                                 tags='day')
                trials = glob.iglob(day + '/*.nd2')
                for trial in trials:
                    trial_parts = trial.split('/')
                    iid = self.file_tree.insert(day_item, 'end',
                                                text=trial_parts[-1],
                                                values=trial,
                                                tags=trial)
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
                    except FileNotFoundError:
                        analysis_status = 'no metadata.yaml file'
                    self.file_tree.item(iid, values=(analysis_status, rating))

            except ValueError:  # we get here if dir name is not a number
                pass  # and we ignore these dirs
        self.file_tree.pack(fill=tk.BOTH, anchor=tk.N)

    def load_metadata_from_yaml(self, metadata_file_path: str) -> dict:
        """Loads metadata from an existing yaml file."""
        with open(metadata_file_path, 'r') as yamlfile:
            metadata = yaml.load(yamlfile)

        return metadata


class AnalyzeTrialFrame(tk.Frame):
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
        self.plot_canvas.get_tk_widget().grid(sticky=tk.W + tk.N + tk.S,
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
        self.part_label_checkbox.state(['selected', '!alternate'])

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

        # Button to calculate strain
        self.calc_strain_button = ttk.Button(
                self.param_frame, text='Calculate strain')
        self.calc_strain_button.grid(row=param_frame_row, column=1,
                                     columnspan=2)
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

        self.explore_data_frame = tk.Frame(self.analysis_notebook)
        self.df_tab = self.analysis_notebook.add(
                self.explore_data_frame, text="Explore Data")
        self.dataframe = None  # TableModel.getSampleData()
        self.dataframe_widget = Table(self.explore_data_frame,
                                      dataframe=self.dataframe)

        self.dataframe_widget.show()


class AnalysisQueueFrame(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.queue_location = ('/Users/adam/Documents/SenseOfTouchResearch/'
                               'SSN_ImageAnalysis/analysis_queue.yaml')
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

        # TODO: update queue when changing tabs on gui
        with open(self.queue_location, 'r') as queue_file:
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


class PlotResultsFrame(tk.Frame):
    def __init__(self, parent, screen_dpi):
        tk.Frame.__init__(self, parent)
        self.parent = parent

        notebook_height = self.parent.winfo_height()
        self.notebook_height_in = notebook_height / screen_dpi

        # Create a figure and a canvas for showing images
        self.fig = mpl.figure.Figure(figsize=(self.notebook_height_in - 1,
                                              self.notebook_height_in - 1))
        self.ax = self.fig.add_axes([0.1, 0.2, 0.8, 0.7])
        self.plot_canvas = FigureCanvasTkAgg(self.fig, self)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().grid(sticky=tk.W + tk.N + tk.S,
                                              row=0,
                                              column=0,
                                              rowspan=3)
        self.plot_canvas.get_tk_widget().grid_rowconfigure(0, weight=1)

        # Creat a frame to put all the controls and parameters in
        self.plot_config_frame = tk.Frame(self)
        config_frame_row = 0

        # Button for analysis progress plots
        self.progress_plot_button = ttk.Button(self.plot_config_frame,
                                               text='Plot analysis progress')
        self.progress_plot_button.grid(row=config_frame_row, column=1)
        config_frame_row += 1

        # Button for strain plot for one trial
        self.plot_strain_one_trial_button = ttk.Button(
                self.plot_config_frame, text='Plot strain- one trial')
        self.plot_strain_one_trial_button.grid(row=config_frame_row, column=1)
        config_frame_row += 1

        self.plot_config_frame.grid(row=0, column=1)

        # TODO: add buttons for interesting plots

    def plot_progress(self, all_statuses_dict: dict, status_values: list):
        status_count = dict.fromkeys(status_values, 0)
        for experiment_id, status in all_statuses_dict.items():
            status_count[status] += 1

        self.ax.bar(range(len(status_count)), status_count.values(),
                    align='center')
        self.ax.set_xticks(range(len(status_count)))
        self.ax.set_xticklabels(list(status_count.keys()),
                                rotation=30, horizontalalignment='right')

        self.plot_canvas.draw()

    def plot_strain_one_trial(self, strain):
        """Plots the strain results from a np.ndarray"""
        self.ax.plot(strain.T)

        self.plot_canvas.draw()

    # TODO: controls for selecting what data are plotted


if __name__ == '__main__':
    import ssn_image_analysis_gui_controller
    controller = ssn_image_analysis_gui_controller.StrainGUIController()
    controller.run()
