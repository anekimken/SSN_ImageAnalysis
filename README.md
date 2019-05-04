# SSN Image Analysis

This is software for calculating mechanical strain in *C. elegans* touch receptor neurons. It uses z-stacks from a spinning disk microscope, finds the fluorescent mitochondria in the touch neurons, and calculates mechanical deformation using the changes in position of the mitochondria. The software was developed and tested on a mac, so it is likely not compatible with Windows.

## Flow of analysis

## Code structure
### MVC

## Data storage structure
The GUI expects data to be stored in a specific way, and will fail to load data properly (or at all) if these conventions aren't implemented properly.

### Raw Data 
The Raw Data from the microscope exists as .nd2 files. This is the standard file type from Nikon microscopes. In principle, any microscope file type could be used, especially if the OME XML is supported. Each file should be named in the format SSN_www_ttt.nd2 (eg SSN_001_001.nd2), where SSN stands for Stress-Strain-Neuron, the moniker we gave these experiments, www is the ID number of the worm in that trial, and ttt is the trial number for that individual worm.

All trials from a given day go in a folder whose name indicates the day those experiments occurred in the form YYYYMMDD (eg 20190503 for May 3, 2019). Importantly, all characters in this directory name must be numbers, or the code will not notice that this is a folder with data for the experiment.

These folders then live together in a folder specified in the 'config.yaml' file. This file should be placed in the base directory of the git repo as noted below in the Installation instructions.

### Analyzed Data
Analyzed data is stored in series of YAML files for each trial. The code generates files for results of particle finding before and after linking, parameters used for each analysis, and results of strain calculations. These files live in a folder specified in the 'config.yaml' file, as noted below in the Installation section.

### Metadata
Initially, metadata is stored in two places - within the .nd2 file with the image data and in a Google spreadsheet. The first time a trial is loaded by the code, it combines the metadata from both sources, makes a single 'metadata.yaml' file, and saves it in the folder dedicated to that trial's analyzed data. 

### Figures
Figures and other plots are mostly made in Jupyter Notebooks to enable interspersed explanations and notes.

## Installation 

The analysis code runs in spyder in a conda environment. Code for making figures is in jupyter notebooks that use the same conda environment.

### Clone the github repository

### Create conda environment using setup_conda_env.yml
This installs all the packages that are needed to run the analysis code and create figures. The `setup_conda_env.yml` file is in the root directory of the git repo. First, [install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) if you haven't already. Then, open a terminal, go to the base directory for the cloned repo, and create the conda environment:
```
conda env create -f setup_conda_env.yml
```
This might take a couple minutes.

### Create a config.yaml file

Open a text editor. Add the text
```
analysis_dir: /path/to/git/repo/
data_dir: /path/to/raw/data/
```
but using the actual paths for the data and code on your computer. Save it with the filename `config.yaml` in the base directory of the repo.

### Launch the conda environment and then spyder or jupyter notebook
The conda environment should automatically be called `SSN_analysis_env`. Launch it in a terminal:
```
source activate SSN_analysis_env
```

Spyder is used for running the data analysis gui. Launch it simply by executing the command:
```
spyder
```

The code that pools together analyzed data into figures is in jupyter notebooks. Launch jupyter by navigating in a terminal to the base directory of the repo and executing the command:
```
jupyter notebook
```

<!---

# Dependencies - probably don't need this because I have env.yml file
## Managed with conda
- tkinter
- glob
- matplotlib
- numpy
- yaml
- warnings
- pandas
- scipy
- math
- pathlib
- time
- pims
- os
- datetime
- typing
- nd2reader
- spread
- oauth2client
- pandastable
- fast_histogram

## Direct from github
- trackpy

-->

