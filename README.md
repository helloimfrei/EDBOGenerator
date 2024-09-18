# EDBOGenerator
An in-progress framework of useful functionality for implementing Bayesian optimization for reaction optimization, built on EDBO+ from the Doyle lab.
https://github.com/doyle-lab-ucla/edboplus

This framework seeks to augment the existing EDBO+ workflow by handling the generation of a run directory and all required files for optimization runs. 
Instantiating the class provides an object representing an entire optimization run for a given reaction scope, with a set of methods to enable optimization/simulation/visualization workflows with built-in file handling.
These methods allow the user to perform optimizations with minimal effort, currently supporting: round-by-round prediction (standard EDBO+ usage), bulk training input (beginning the optimization run with existing experimental data), simulation of an optimization using existing experimental data, optimization summary generation (coming soon!), and interactive visualization (coming soon!).

Planned features include a Shiny-based UI to enable user-friendly data input (my goal is to fully negate the need to open a csv when using this framework)

## Installation:
1. Get EDBO running locally using the instructions highlighted in the EDBOPlus repo linked above.
2. Install edbogenerator with pip or by cloning this repo:
    ```bash
    pip install edbogenerator
    ```

    or

    ```bash
    git clone https://github.com/helloimfrei/EDBOGenerator.git
    cd EDBOGenerator/edbogenerator/
    pip install .
    ```
4. Check out the examples folder for a demo workflow!


