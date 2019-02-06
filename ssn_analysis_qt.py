#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 15:29:22 2019

@author: adam
"""

# import matplotlib as mpl
# import numpy as np
# from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk
import glob
import yaml
import numpy as np
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import matplotlib.pyplot as plt

import strain_propagation_trial
from PandasModel import PandasModel


def select_trial():
    """Shows for selecting the trial to analyze and show info about it"""
    trial = strain_propagation_trial.StrainPropagationTrial()
    root = tk.Tk()
#    notebook = ttk.Notebook(root)

    file_load_frame = FileLoadFrame(root)
#    file_load_tab = notebook.add(
#        file_load_frame, text="Load File(s)")

    # pack notebook and update so notebook has the right size
#    notebook.pack(expand=1, fill=tk.BOTH)
    file_load_frame.pack(expand=1, fill=tk.BOTH)
    root.update()

    root.title("Choose a file")
    file_load_frame.mainloop()

    filename = file_load_frame.file_to_analyze.get()
    trial.load_trial(filename, load_images=True, overwrite_metadata=False)

    return trial


def display_images(trial):
    """Displays images of the trial and any results that exist"""
    # Use pyqtgraph for analyzing data

    # Interpret image data as row-major instead of col-major
    pg.setConfigOptions(imageAxisOrder='row-major')

    app = QtGui.QApplication([])

    # Create window with ImageView widget
    win = QtGui.QMainWindow()
    win.resize(800, 800)
    imv = pg.ImageView()
    win.setCentralWidget(imv)
    win.show()
    win.setWindowTitle('ImageView of Current Trial')

    # Display the max projection
    images_to_display = np.amax(trial.image_array, 1)  # collapse z axis
    images_to_display = images_to_display.squeeze()
    imv.setImage(images_to_display)

    # Set a custom color map
#    colors = [
#        (0, 0, 0),
#        (45, 5, 61),
#        (84, 42, 55),
#        (150, 87, 60),
#        (208, 171, 141),
#        (255, 255, 255)
#    ]
    cmap = generatePgColormap('magma')
#    cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
    imv.setColorMap(cmap)

    app.exec_()


def generatePgColormap(cm_name):
    pltMap = plt.get_cmap(cm_name)
    colors = pltMap.colors
    colors = [c + [1.] for c in colors]
    positions = np.linspace(0, 1, len(colors))
    pgMap = pg.ColorMap(positions, colors)
    
    return pgMap


def enqueue_trial():
    """Adds a trial to the queue to be processed tonight"""
    pass


class FileLoadFrame(tk.Frame):
    """This class implements the first tab of the GUI, which prompts the
    user to choose which file or files to analyze
    """
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent

        self.file_to_analyze = tk.StringVar()

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
        self.file_tree.bind(
                '<<TreeviewSelect>>', func=self.on_file_selection_changed)

    def update_file_tree(self):
        # now, we load all the file names into a Treeview for user selection
        data_location = '/Users/adam/Documents/SenseOfTouchResearch/SSN_data/*'
        experiment_days = glob.iglob(data_location)

        self.file_tree = tk.ttk.Treeview(self, height=37,
                                         columns=("status", "filename"))
        self.file_tree.heading("status", text="Analysis Status")
        self.file_tree.heading("filename", text="Full file path")
        # TODO: show number of stars for analysis here instead of file path
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
        self.file_tree.pack(fill=tk.BOTH, anchor=tk.N)

    def on_file_selection_changed(self, event):
        """This function keeps track of which files are selected for
        analysis. If more than one files are selected, the Run Batch option
        is available, otherwise only the Load Trial button is active.
        """
        file_tree = self.file_tree
        self.file_list = []
        selection = file_tree.selection()
        for this_item in selection:
            info = file_tree.item(this_item)
            self.file_list.append(info['values'])
        if len(self.file_list) > 1:
            self.load_trial_button.config(state=tk.DISABLED)
            self.run_multiple_files_button.config(state=tk.NORMAL)
        elif len(self.file_list) == 1 and info['tags'] == ['day']:
            self.load_trial_button.config(state=tk.DISABLED)
            self.run_multiple_files_button.config(state=tk.NORMAL)
            # TODO: add all trials from this day to file_list
        elif len(self.file_list) == 1 and info['tags'] == ['trial']:
            self.load_trial_button.config(state=tk.NORMAL)
            self.run_multiple_files_button.config(state=tk.DISABLED)

        self.file_to_analyze.set(self.file_list[0][1])
        print(self.file_list[0][1])

    def load_metadata_from_yaml(self, metadata_file_path: str) -> dict:
        """Loads metadata from an existing yaml file."""
        with open(metadata_file_path, 'r') as yamlfile:
            metadata = yaml.load(yamlfile)

        return metadata


trial = select_trial()
display_images(trial)
