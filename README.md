# SSN Image Analysis

This is software for calculating mechanical strain in *C. elegans* touch receptor neurons. It uses z-stacks from a spinning disk microscope, finds the fluorescent mitochondria in the touch neurons, and calculates mechanical deformation using the changes in position of the mitochondria. The software was developed and tested on a mac, so it is likely not compatible with Windows.

## Code structure
### MVC

## Data storage structure
### Raw Data
### Metadata
### Analyzed Data
### Figures

## Installation 

The analysis code runs in spyder in a conda environment. Code for making figures is in jupyter notebooks that use the same conda environment.

### Clone the github repository

### Create conda environment using setup_conda_env.yml
This installs all the packages that are needed to run the analysis code and create figures. The `setup_conda_env.yml` file is in the root directory of the git repo. First, [install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) if you haven't already. Then, open a terminal and create the conda environment:
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

