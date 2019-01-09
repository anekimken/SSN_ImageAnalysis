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
from tkinter import ttk
import glob
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import strain_propagation_trial as ssn_trial


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
        self.root.title("SSN Image Analysis")
        w, h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry("%dx%d+0+0" % (w, h))
        self.notebook = ttk.Notebook(root)

        self.file_load_frame = FileLoadFrame(self.notebook)
        self.notebook.add(self.file_load_frame, text="Load File(s)")

        self.inspect_image_frame = InspectionFrame(self. notebook)
        self.notebook.add(self.inspect_image_frame, text="Inspect Images")

        self.test_parameter_frame = ParamTestFrame(self.notebook)
        self.notebook.add(self.test_parameter_frame,
                          text="Test Analysis Parameters")

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

        # define buttons for starting analysis
        self.button_frame = tk.Frame(self)
        self.load_trial_button = tk.Button(self.button_frame,
                                           text="Load Trial",
                                           state=tk.DISABLED,
                                           command=self.load_trial)
        self.load_trial_button.pack(side=tk.RIGHT)
        self.run_batch_button = tk.Button(self.button_frame, text="Run Batch",
                                          state=tk.DISABLED)
        self.run_batch_button.pack(side=tk.RIGHT)
        self.button_frame.pack(side=tk.BOTTOM)
        # TODO: implement button responses (go to next tab and get started)

        # now, we load all the file names into a Treeview for user selection
        data_location = '/Users/adam/Documents/SenseOfTouchResearch/SSN_data/*'
        experiment_days = glob.iglob(data_location)
        self.file_tree = tk.ttk.Treeview(self, height=37,
                                         columns=("filename", "status"))
        self.file_tree.heading("filename", text="Full file path")
        self.file_tree.heading("status", text="Analysis Status")
        for day in experiment_days:
            try:
                int(day[-8:])  # check to make sure this dir is an experiment
                day_item = self.file_tree.insert('', 'end', text=day[-8:],
                                                 tags='day')
                trials = glob.iglob(day + '/*.nd2')
                for trial in trials:
                    trial_parts = trial.split('/')
                    self.file_tree.insert(day_item, 'end',
                                          text=trial_parts[-1],
                                          values=trial,
                                          tags='trial')
                    # TODO: get analysis status and add to table
            except ValueError:  # we get here if dir name is not a number
                pass  # and we ignore these dirs
        self.file_tree.bind('<<TreeviewSelect>>',
                            self.on_file_selection_changed)
        self.file_tree.pack(fill=tk.BOTH, anchor=tk.N)  # fill = tk.X,

        # TODO: add checkbox for overwriting metadata

    def load_trial(self):
        """Loads the data for this trial from disk and sends us to the
        inspection tab to start the analysis
        """
        # OPTIMIZE: load trial in a separate thread
        self.trial = ssn_trial.StrainPropagationTrial(self.file_list[0][0])

    def on_file_selection_changed(self, event):
        """This function keeps track of which files are selected for
        analysis. If more than one files are selected, the Run Batch option
        is available, otherwise only the Load Trial button is active.
        """
        self.file_list = []
        selection = self.file_tree.selection()
        for this_item in selection:
            info = self.file_tree.item(this_item)
            self.file_list.append(info['values'])
        if len(self.file_list) > 1:
            self.load_trial_button.config(state=tk.DISABLED)
            self.run_batch_button.config(state=tk.NORMAL)
        elif len(self.file_list) == 1 and info['tags'] == ['day']:
            self.load_trial_button.config(state=tk.DISABLED)
            self.run_batch_button.config(state=tk.NORMAL)
            # TODO: add all trials from this day to file_list
        elif len(self.file_list) == 1 and info['tags'] == ['trial']:
            self.load_trial_button.config(state=tk.NORMAL)
            self.run_batch_button.config(state=tk.DISABLED)
        # print(self.file_list)


class InspectionFrame(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        # label = tk.ttk.Label(self,text="check me out?")
        # label.grid(column=1, row=1)
        # label.pack()

        # TODO: add canvas for showing image

        # test dater tots
        X = np.linspace(0, 2 * np.pi, 50)
        Y = np.sin(X)

        # Create the figure we desire to add to a canvas
        fig = mpl.figure.Figure(figsize=(1, 2))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.plot(X, Y)

        # create the canvas with our figure
        self.plot_canvas = FigureCanvasTkAgg(fig, self)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)

        # TODO: add controls for slice, timepoint, max projection

        # TODO: add text showing stack height and metadata
        
        # TODO: add button for loading images


class ParamTestFrame(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        label = tk.ttk.Label(self, text="try me out?")
        label.grid(column=1, row=1)
        label.pack()

        # TODO: add canvas for showing results of one frame

        # TODO: add input fields for parameters

        # TODO: add button to start analysis

        # TODO: add text when finished to indicate time and relevant results


class AnalyzeTrialFrame(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        label = tk.ttk.Label(self, text="run me?")
        label.grid(column=1, row=1)
        label.pack()

        # TODO: add canvas for showing results of one frame

        # TODO: add input fields for parameters

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


root = tk.Tk()
analysisGUI = SSN_analysis_GUI(root)
# TODO: bring new window to top
root.mainloop()
