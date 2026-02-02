# Multi-channel wireless sensing 

## Description
The project contains modules to analyize data from different instruments. The data is in the form of electrical impedance (magnitude, phase, L, C, R) or reflection coefficient (S11 magnitude, phase). 
The goal of the project is to analyze data from a multisensors wireless sensing platform featuring one textile inductor (LS) connected to multiple capacitive sensors. 
The readout is done through inductive coupling between LS and an external reader inductor (LR) connected to the measuring instrument. 
Capacitors are either commercial components or textile based sensors. 


## Environment Setup
To ensure reproducibility, this project includes both:
- environment.yml → Full Conda environment (recommended)
- requirements.txt → Lightweight Pip alternative
The environment can be recreated with either method as described below.
option 1 (yml file): in a command window, use the following commands
    conda env create -f environment.yml
    conda activate myproject
option 2 (txt file): in a command window, use the following commands
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    pip install -r requirements.txt


## Usage
Environment Setup
The modules in the ''utils'' folder provide utilities to read data, make calculations or plot data:
- classification.py (specific for classification task of activities)
- interactive.py: functions for interactively choosing directories and files to analyze
- plot.py (general plots, includes classification plots)
- read.py: read data from measurements (impedance analyzer, VNA, nanoVNA, etc)
- utilities.py: general utility functions, calculations, etc

All other modules are used to analyze and plot data from the different experimental measurements. 
The module name may contain the measurement instrument (e.g., ''IA'' for impedance analyzer). 
For measurements on single components (e.g., capacitive sensor or textile inductor), the appendix ''Z_single_components'' is used:
- IA_Z_single_components.py (IA: impedance analyzer)
- LCR_Z_single_components.py (LCR: LCR meter)

Data analysis of results from bench test is performed with the module:
- VNA_bench_tests.py (2 capacitive sensors, benchtop VNA)

Data analysis of results from tests with sensorized prototype (activity tests, e.g. 
squat, walk, run, etc) is performed with the modules:
- VNA_activities_preprocess.py (converts s1p files to csv files with all resonance frequencies in time)
- VNA_activities_classify.py (labels data - standing vs activity phases - and extract features, then trains ML classification models)
- VNA_activitieis_predict.py (predicts on separate test dataset using previously trained model)


## Directories structure
projectname/
│
├── <meas_instr1>            # IA, LCR, or VNA
├── ├── data/                # Data
├── ├── ├── <test_group1>      # name of the test, e.g. "bench_test1". Data for test_group1
├── ├── ├── <test_group2>      # name of the test, e.g. "bench_test2" . Data for test_group1
├── ├── ├── ...
├── ├── results/             # Results (results of preprocessing and analysis, mostly csv files)
├── ├── ├── <test_group1>      # Results for test_group1 
├── ├── ├── ...
├── ├── plots/               # Plots 
├── ├── ├──<test_group1>      # Plots for test_group1 
├── ├── ├── ...
├── ├── classification       # NB: only for activities data, for meas_instr=VNA
├── <meas_instr2>   
├── ...
├── code                 # Source code (models, preprocessing, utils, etc.)
├──├── environment.yml      # Conda environment definition
├──├── requirements.txt     # Pip environment definition
├──├── main.py              # Main entry point
└──├── README.md            # This file