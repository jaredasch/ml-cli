## What is this?
This is a tool to run some ML subroutines from the command line. I'm primarly using this as a learning experience for C++ and how to optimize the implementation of ML algorithms.

## Instructions to Run on Ubuntu
- Install the Armadillo library (used for linear algebra) with the following terminal commands
    - `sudo apt-get install libblas-dev libopenblas-dev liblapack-dev libarpack2-dev libsuperlu-dev`
    - `sudo apt-get install libarmadillo-dev`
- Run `make` in the root directory, and then call `ml-cli`