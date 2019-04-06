# SSN Image Analysis

This is software for calculating mechanical strain in *C. elegans* touch receptor neurons. It uses z-stacks from a spinning disk microscope, finds the fluorescent mitochondria in the touch neurons, and calculates mechanical deformation using the changes in position of the mitochondria. The software was developed and tested on a mac, so it is likely not compatible with Windows.

## Installation 

The analysis code runs in spyder in a conda environment. Code for making figures is in jupyter notebooks that use the same conda environment.

### Create conda environment from environment.yml
This installs all the packages that are needed to run the analysis code and create figures. First, [install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) if you haven't already. Then, open a terminal and create the conda environment:
```
conda env create -f environment.yml
```
This might take a couple minutes.

### Clone the github repository

### Modify file locations in config.yml

Open config.yml in a text editor. It should be in the base directory of the cloned repo.

### Launch the conda environment and then spyder or jupyter notebook
The conda environment should automatically be called SSN_analysis_env. Launch it in a terminal:
'''
source activate SSN_analysis_env
'''

Spyder is used for running the data analysis gui. Launch it simply by executing the command:
'''
spyder
'''

The code that pools together analyzed data into figures is in jupyter notebooks. Launch jupyter by navigating in a terminal to the directory where the repo is cloned and executing the command:
'''
jupyter notebook
'''

## Data storage structure
### Raw Data
### Metadata
### Analyzed Data
### Figures

<!---
your comment goes here
and here
-->

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




## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

