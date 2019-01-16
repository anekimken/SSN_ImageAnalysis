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
        # self.notebook.height_in = 4

        self.file_load_frame = FileLoadFrame(self.notebook)
        self.file_load_tab = self.notebook.add(
                self.file_load_frame, text="Load File(s)")

        # pack notebook and update so notebook has the right size
        self.notebook.pack(expand=1, fill=tk.BOTH)
        self.root.update()

        self.inspect_image_frame = InspectionFrame(self.notebook, self.dpi)
        self.notebook.add(self.inspect_image_frame, text="Inspect Images")

        self.analyze_trial_frame = AnalyzeTrialFrame(self.notebook)
        self.notebook.add(self.analyze_trial_frame, text="Analyze Trial")

        self.plot_results_frame = PlotResultsFrame(self.notebook)
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
        self.run_batch_button = tk.Button(self.button_frame, text="Run Batch",
                                          state=tk.DISABLED)
        self.run_batch_button.pack(side=tk.RIGHT)
        self.button_frame.pack(side=tk.BOTTOM)

        # now, we load all the file names into a Treeview for user selection
        data_location = '/Users/adam/Documents/SenseOfTouchResearch/SSN_data/*'
        experiment_days = glob.iglob(data_location)
        # self.analyzed_data_location = pathlib.Path(
        #         '/Users/adam/Documents/SenseOfTouchResearch/'
        #         'SSN_ImageAnalysis/AnalyzedData/' + self.experiment_id + '/')
        # self.metadata_file_path = self.analyzed_data_location.joinpath(
        #          'metadata.yaml')
        # self.param_test_history_file = self.analyzed_data_location.joinpath(
        #          'trackpyParamTestHistory.yaml')

        self.file_tree = tk.ttk.Treeview(self, height=37,
                                         columns=("status", "filename"))
        self.file_tree.heading("status", text="Analysis Status")
        self.file_tree.heading("filename", text="Full file path")
        for day in experiment_days:
            # print(day)
            # metadata_location = (data_location
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
                                                tags='trial')
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
                    except FileNotFoundError:
                        analysis_status = 'no metadata.yaml file'
                    self.file_tree.item(iid, values=(analysis_status, trial))

            except ValueError:  # we get here if dir name is not a number
                pass  # and we ignore these dirs
        self.file_tree.pack(fill=tk.BOTH, anchor=tk.N)  # fill = tk.X,

    def load_metadata_from_yaml(self, metadata_file_path: str) -> dict:
        """Loads metadata from an existing yaml file."""
        with open(metadata_file_path, 'r') as yamlfile:
            metadata = yaml.load(yamlfile)

        return metadata


class InspectionFrame(tk.Frame):
    def __init__(self, parent, screen_dpi):
        # TODO: docstring
        tk.Frame.__init__(self, parent)
        self.parent = parent

        # get size for making figure
        notebook_height = self.parent.winfo_height()
        self.notebook_height_in = notebook_height / screen_dpi

        # Create a figure and a canvas for showing images
        self.fig = mpl.figure.Figure(figsize=(self.notebook_height_in / 2,
                                              self.notebook_height_in))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.plot_canvas = FigureCanvasTkAgg(self.fig, self)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().grid(sticky=tk.W + tk.N + tk.S,
                                              row=0,
                                              column=0,
                                              rowspan=3)
        self.plot_canvas.get_tk_widget().grid_rowconfigure(0, weight=1)

        # Creat a frame to put all the controls and parameters in
        self.param_frame = tk.Frame(self)
        param_frame_row = 0

        # Optional max projection
        self.max_proj_checkbox_status = tk.IntVar(value=1)
        self.max_proj_checkbox = ttk.Checkbutton(
                self.param_frame,
                text='Max projection',
                variable=self.max_proj_checkbox_status)
        self.max_proj_checkbox.grid(
                row=param_frame_row, column=1, columnspan=2)
        self.max_proj_checkbox.state(['selected', '!alternate'])
        param_frame_row += 1

        # Select slice to show
        self.slice_selector_label = ttk.Label(self.param_frame,
                                              text='Choose slice to display: ')
        self.slice_selector_label.grid(row=param_frame_row, column=1)
        self.slice_selector = tk.Spinbox(self.param_frame, values=(1, 2))
        self.slice_selector.grid(row=param_frame_row, column=2)
        param_frame_row += 1

        # Select timepoint to show
        self.timepoint_selector_label = ttk.Label(
                self.param_frame, text='Choose timepoint to display: ')
        self.timepoint_selector_label.grid(row=param_frame_row, column=1)
        self.timepoint_selector = tk.Spinbox(self.param_frame, values=(1, 2))
        self.timepoint_selector.grid(row=param_frame_row, column=2)
        param_frame_row += 1

        # Button for toggling mito candidate labels image
        self.plot_labels_box_status = tk.IntVar(value=1)
        self.plot_labels_box = ttk.Checkbutton(
                self.param_frame,
                text='Plot mitochondria candidates?',
                variable=self.plot_labels_box_status)
        self.plot_labels_box.grid(
                row=param_frame_row, column=1, sticky=tk.N)
        self.plot_labels_box.state(['selected', '!alternate'])
        self.param_frame.grid_rowconfigure(param_frame_row, minsize=50)

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
        self.btm_slice_selector = tk.Spinbox(self.param_frame, values=(1, 2))
        self.btm_slice_selector.grid(row=param_frame_row, column=2)
        param_frame_row += 1

        # Selector for top slice
        self.top_slice_label = ttk.Label(self.param_frame,
                                         text='Top slice: ')
        self.top_slice_label.grid(row=param_frame_row, column=1)
        self.top_slice_selector = tk.Spinbox(self.param_frame, values=(1, 2))
        self.top_slice_selector.grid(row=param_frame_row, column=2)
        param_frame_row += 1

        # gaussianWidth,
        self.gaussian_blur_width_label = ttk.Label(
                self.param_frame, text='Width of Gaussian blur kernel: ')
        self.gaussian_blur_width_label.grid(row=param_frame_row, column=1)
        self.gaussian_blur_width = tk.Spinbox(self.param_frame,
                                              values=list(range(0, 10)))
        self.gaussian_blur_width.grid(row=param_frame_row, column=2)
        param_frame_row += 1

        # particleZDiameter,
        self.z_diameter_label = ttk.Label(self.param_frame,
                                          text='Particle z diameter: ')
        self.z_diameter_label.grid(row=param_frame_row, column=1)
        self.z_diameter_selector = tk.Spinbox(self.param_frame,
                                              values=list(range(1, 45, 2)))
        self.z_diameter_selector.grid(row=param_frame_row, column=2)
        param_frame_row += 1

        #                 particleXYDiameter,
        self.xy_diameter_label = ttk.Label(self.param_frame,
                                           text='Particle xy diameter: ')
        self.xy_diameter_label.grid(row=param_frame_row, column=1)
        self.xy_diameter_selector = tk.Spinbox(self.param_frame,
                                               values=list(range(1, 45, 2)))
        self.xy_diameter_selector.grid(row=param_frame_row, column=2)
        param_frame_row += 1

        #                 brightnessPercentile,
        self.brightness_percentile_label = ttk.Label(
                self.param_frame, text='Brightness percentile: ')
        self.brightness_percentile_label.grid(row=param_frame_row, column=1)
        self.brightness_percentile_selector = tk.Spinbox(
                self.param_frame, values=list(range(1, 100)))
        self.brightness_percentile_selector.grid(row=param_frame_row, column=2)
        param_frame_row += 1

        #                 minParticleMass,
        self.min_particle_mass_label = ttk.Label(
                self.param_frame, text='Minimum particle mass: ')
        self.min_particle_mass_label.grid(row=param_frame_row, column=1)
        self.min_particle_mass_selector = tk.Entry(self.param_frame)
        self.min_particle_mass_selector.grid(row=param_frame_row, column=2)
        param_frame_row += 1

        # Button to move to next tab
        self.test_param_button = ttk.Button(
                self.param_frame, text='Start parameter testing')
        self.test_param_button.grid(
                row=param_frame_row, column=1, columnspan=2)

        self.param_frame.grid(row=0, column=1)

        # Frame for saving data etc.
        self.save_data_frame = tk.Frame(self)
        save_frame_row = 0

        self.metadata_notes_label = ttk.Label(self.save_data_frame,
                                              text='Notes from metadata:')
        self.metadata_notes_label.grid(row=save_frame_row, column=1)
        save_frame_row += 1

        self.metadata_notes = tk.Message(self.save_data_frame)
        self.metadata_notes.grid(row=save_frame_row, column=1)
        save_frame_row += 1

        self.stack_height_label = ttk.Label(self.save_data_frame,
                                            text='Stack height:')
        self.stack_height_label.grid(row=save_frame_row, column=1)
        save_frame_row += 1

        self.neuron_id_label = ttk.Label(self.save_data_frame,
                                         text='Neuron:')
        self.neuron_id_label.grid(row=save_frame_row, column=1)
        save_frame_row += 1

        self.vulva_side_label = ttk.Label(self.save_data_frame,
                                          text='Vulva side:')
        self.vulva_side_label.grid(row=save_frame_row, column=1)
        save_frame_row += 1

        self.status_dropdown = ttk.Combobox(
                self.save_data_frame,
                state="readonly",
                values=[
                        'No metadata.yaml file',
                        'Not started',
                        'Testing parameters',
                        'Testing parameters for batch',
                        'Strain calculated'])
        self.status_dropdown.grid(column=1, row=save_frame_row)
        save_frame_row += 1

        self.save_data_frame.grid(row=0, column=2)

        # TODO: add button for loading images that aren't yet loaded


class AnalyzeTrialFrame(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        label = tk.ttk.Label(self, text="run me?")
        label.grid(column=1, row=1)
        label.pack()

        # TODO: add canvas for showing results of one frame

        # TODO: add input fields for parameters
        #                 trackingSeachRadius
        # self.top_slice_label = ttk.Label(self.param_frame,
        #                                  text='Tracking search radius: ')
        # self.top_slice_label.grid(row=param_frame_row, column=1)
        # self.top_slice_selector = tk.Spinbox(self.param_frame, values=(1, 2))
        # self.top_slice_selector.grid(row=param_frame_row, column=2)
        # param_frame_row += 1

        # TODO: add button to start analysis

        # TODO: add text when finished to indicate time and relevant results


class PlotResultsFrame(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        label = tk.ttk.Label(self, text="check out my science?")
        label.grid(column=1, row=1)
        label.pack()

        # TODO: add canvas for showing results of one frame

        # TODO: add buttons for interesting plots


if __name__ == '__main__':
    import SSN_image_analysis_GUI
    controller = SSN_image_analysis_GUI.StrainGUIController()
    controller.run()
