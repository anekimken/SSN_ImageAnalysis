#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:32:39 2019

@author: Adam Nekimken
"""


import tkinter as tk
from tkinter import ttk
import glob

class SSN_analysis_GUI(tk.Frame):
    """
    This class implements a GUI for performing all parts of the image analysis
    process and plots the results.
    """
    def __init__(self,root):
        """
        Initializes the Analysis gui object
        """
        tk.Frame.__init__(self,root)
        self.root = root
        self.root.title("SSN Image Analysis")
        w, h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry("%dx%d+0+0" % (w, h))
        notebook = ttk.Notebook(root)
        
        # self.file_load_frame = tk.ttk.Frame(self.notebook)
        # self.notebook.add(self.file_load_frame,text="Load Files")
        # label = tk.ttk.Label(self.file_load_frame,text="Choose file(s)")
        # label.pack()
        # data_location = '/Users/adam/Documents/SenseOfTouchResearch/SSN_data/*'
        # experiment_days = glob.iglob(data_location)
        # self.file_tree = tk.ttk.Treeview(self.file_load_frame,columns=("A","B"))
        # for day in experiment_days:
        #     try:
        #         int(day[-8:]) # check to make sure this dir is an experiment
        #         day_iid = self.file_tree.insert('','end',text =day[-8:])
        #         trials = glob.iglob(day + '/*.nd2')
        #         for trial in trials:
        #             self.file_tree.insert(day_iid,'end',text = trial)
        #     except ValueError:
        #         pass
        # self.file_tree.pack(expand = 1, fill = tk.X, anchor=tk.N)
        
        file_load_frame = FileLoadFrame(notebook)
        # self.file_load_frame.pack()
        notebook.add(file_load_frame,text="Load File(s)")
        
        
        # self.inspect_image_frame = tk.ttk.Frame(self.notebook)
        # self.notebook.add(self.inspect_image_frame,text="Inspect Images")
        # label = tk.ttk.Label(self.inspect_image_frame,text="check me out?")
        # label.grid(column=1, row=1)
        
        # self.test_parameter_frame = tk.ttk.Frame(self.notebook)
        # self.notebook.add(self.test_parameter_frame,
        #                   text="Test Analysis Parameters")
        # label = tk.ttk.Label(self.test_parameter_frame,text="try me out?")
        # label.grid(column=1, row=1)
        
        # self.analyze_trial_frame = tk.ttk.Frame(self.notebook)
        # self.notebook.add(self.analyze_trial_frame,text="Analyze Trial")
        # label = tk.ttk.Label(self.analyze_trial_frame,text="Run me!")
        # label.grid(column=1, row=1)
        
        # self.plot_results_frame = tk.ttk.Frame(self.notebook)
        # self.notebook.add(self.plot_results_frame,text="Plots Results")
        # label = tk.ttk.Label(self.plot_results_frame,text="See my science!")
        # label.grid(column=1, row=1)
        
        notebook.pack(expand = 1,fill=tk.BOTH)

class FileLoadFrame(tk.Frame):       
    def __init__(self,parent):
        tk.Frame.__init__(self,parent)
        # self.parent = parent
        # self.file_load_frame = tk.ttk.Frame(self.notebook)
        # self.notebook.add(self.file_load_frame,text="Load Files")
            
        self.label = tk.ttk.Label(self,text="Choose file(s)")
        self.label.pack()
        data_location = '/Users/adam/Documents/SenseOfTouchResearch/SSN_data/*'
        experiment_days = glob.iglob(data_location)
        self.file_tree = tk.ttk.Treeview(self,columns=("A","B"))
        for day in experiment_days:
            try:
                int(day[-8:]) # check to make sure this dir is an experiment
                day_iid = self.file_tree.insert('','end',text =day[-8:])
                trials = glob.iglob(day + '/*.nd2')
                for trial in trials:
                    self.file_tree.insert(day_iid,'end',text = trial)
            except ValueError:
                pass
        self.file_tree.pack(expand = 1, fill = tk.X, anchor=tk.N)
        
    
    
root = tk.Tk()
analysisGUI = SSN_analysis_GUI(root)
root.mainloop()
# analysisGUI.run()
# analysisGUI.root.destroy()

