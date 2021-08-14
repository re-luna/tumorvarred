# tumorvarred

Python code of the Master thesis "Mathematical modelling and simulation for tumour growth and angiogenesis".

# Installation

For installation is recommended to start with a fresh new environment for FiPy. See FiPy [installation](https://www.ctcms.nist.gov/fipy/INSTALLATION.html).

1. Using conda or miniconda run:
```bash
$ conda create --name <MYFIPYENV> --channel conda-forge python=<PYTHONVERSION> fipy
```

2. Activate the new environment:
```bash
$ conda activate <MYFIPYENV>
```

3. Install the following packages on the new environment <MYFIPYENV>:
```bash
$ conda install numpy
$ conda install scipy
$ conda install matplotlib
```

# Usage

The input files are:
- default.ini,
it contains all data tables used in the thesis. It contains 3 sections [main], [environment] and [abm].

- blood vessels.xml, contains the information of inflow (pressure, position), outflow nodes (pressure, position) and connections.

## default file
The project contains two examples:

- staticvasculature.ini, contains the values for the simulations of Chapter 3.

- angiognesis.ini, contains the values for the simulations of Chapter 5.

Change the name of one of these files for "default.ini" to choose the desired simulation.

### Simulation options in default.ini ​
Domain size, scale, period of simulation and stability.

```
# Size of the lattice (lx,ly)
LX = 50
LY = 50

# Parameters for the multiscale simulation, dt in minutes
STEPS = 1920
DT = 30
SPATIAL_DELTA = 0.004
CFL = 1.9
```
LX and LY are the number of grid cells. SPATIAL_DELTA is the size of every grid cell in cm.

DT is the size of every time step in minutes. STEPS is the number of time steps that the simulation lasts.

CFL is the constant for CFL condition computation. The higher the faster but may leads to instability.


Plotting and saving options.
```
# Save path and plotting options
SHOW_PLOT = 0  #Shows the plot of the simulation every SAVE_STEPS 
SAVE_PLOT = 0  #Save the states of abm, oxygen, VEGF and cell number densities every SAVE_STEPS
SAVE_PATH = result/default
SAVE_NAME = plot_
SAVE_STEPS = 1 
```
Notice: After the simulation ends all the values of the simulation will be saved. Use SAVE_PLOT for debugging only.


## blood vessels.xml
This file contains the inflow nodes, outflow nodes and the connection of them.
```xml
<?xml version="1.0"?>
<data>
    <node> #this is the node 0
        <x>-1.0</x>   # This is the position x of the node 0
        <y>12.0</y>   # This is the position y of the node 0
        <p>-25.0</p>  # This is the pressure of the node 0
    </node>
    <node> #this is the node 1
        <x>50.0</x>
        <y>12.0</y>
        <p>10.0</p>
    </node>
    <node> #this is the node 2
        <x>-1.0</x>
        <y>37.0</y>
        <p>-25.0</p>
    </node>
    <node> #this is the node 3
        <x>50.0</x>
        <y>37.0</y>
        <p>10.0</p>
    </node>
    <connection> # Connection of the blood vessel 0
        <in_node>0</in_node>   #Here we indicate that node 0 is the inflow node of the blood vessel 0
        <out_node>1</out_node> #Here we indicate that node 1 is the outflow node of the blood vessel 0
    </connection>
    <connection> # Connection of the blood vessel 1
        <in_node>2</in_node>   #Here we indicate that node 2 is the inflow node of the blood vessel 1
        <out_node>3</out_node> #Here we indicate that node 3 is the inflow node of the blood vessel 1
    </connection>
</data>
```

Note that, the inflow nodes must have a negative value of pressure (For assembling the linear system).

# Result of the simulation
The simulation gives as final result two files:
- density_data.obj, contains all the data of the simulation in array form. Pickle is used for this purpose.
- log.log, contains the log of the run in text format.

The code also can provide csv files, use SAVE_PLOT for this purpose.

## Reduced variance algorithm
The reduced variance algorithm is implemented using a jupyter notebook called "Data density analysis.ipynb". The file is located in the "result" directory.

Once several runs are performed (at least 10), run "Data density analysis.ipynb".

In "Data density analysis.ipynb".
```
FOLDERPATH = 'default'
SAVEPATH = 'average/'

SAMPLES = 10
STEP_SIZE = 30
STEP_Q = 1920
STEP_LAST = 1919
```
SAMPLES is the number of successful runs performed. Note that, the results must be enumerated and named from default0 to default9 as example.

STEP_SIZE is the time step size, STEP_Q is the quantity of time steps simulated and STEP_LAST is the last time step to be analyzed.

Important: STEP_SIZE and STEP_Q must coincide with the input in default.ini.

The result of the analysis is saved in the folder result/average.

## Video of simulations
Examples of numerical simulations:
    
Simulation of reduced variance: https://www.youtube.com/watch?v=PgKsiV-DO3
Simulation with angiogenesis:  https://www.youtube.com/watch?v=pFa0oxxwt7k
