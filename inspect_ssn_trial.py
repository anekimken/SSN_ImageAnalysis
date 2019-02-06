#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 17:07:35 2019

@author: adam
"""

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk
import glob
import yaml

import strain_propagation_trial
from strain_propagation_view import FileLoadFrame


def main():
    trial = strain_propagation_trial.StrainPropagationTrial()
    root = tk.Tk()
    notebook = ttk.Notebook(root)

    file_load_frame = FileLoadFrame(notebook)
    file_load_tab = notebook.add(
        file_load_frame, text="Load File(s)")

    # pack notebook and update so notebook has the right size
    notebook.pack(expand=1, fill=tk.BOTH)
    root.update()

    root.title("Choose a file")
    root.mainloop()

    filename = ('/Users/adam/Documents/SenseOfTouchResearch/SSN_data/20190204/'
                'SSN_152_001.nd2')
    
    trial.load_trial(filename, load_images=True, overwrite_metadata=True)
    
    my_plot = plt.imshow(trial.image_array[1, 20, :, :], interpolation='nearest')
    
    
    # max projection
    stack = trial.image_array[1]
    image_to_display = np.amax(stack, 0)  # collapse z axis
    image_to_display = image_to_display.squeeze()
    max_projection = plt.imshow(trial.image_array[1, 20, :, :],
                                interpolation='nearest')
    
    
    # # single slice
    # image_to_display = self.trial.image_array[
    # selected_timepoint, selected_slice]

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

    def load_metadata_from_yaml(self, metadata_file_path: str) -> dict:
        """Loads metadata from an existing yaml file."""
        with open(metadata_file_path, 'r') as yamlfile:
            metadata = yaml.load(yamlfile)

        return metadata

main()
