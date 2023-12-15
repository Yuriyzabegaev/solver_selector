# Automated solver selection for simulation of multiphysics processes in porous media


## Installation
A Docker image with the full environment to reproduce the experiments, including PorePy and PETSc, is available on Zenodo (TODO).
This is how to access it ([Docker](https://www.docker.com/) should be installed):
```
docker run -dit --name solver_selector -p 8888:8888 solver_selector:latest
docker exec -it solver_selector /bin/bash
```
Note that `-p 8888:8888` is a port forwarding to make the Jupyter Notebooks server accessible from your browser to visualize the results of the experiments.

## Alternative installation
It is recommended to use the Docker image. However, this is a recipe of manual installation.
1. Install [PETSc](https://petsc.org/) with Hypre and PETSc4Py.
2. Install [PorePy](https://github.com/pmgbergen/porepy) from the `develop` branch.
5. Clone this repository:
```
git clone https://github.com/Yuriyzabegaev/solver_selector.git
cd solver_selector
```
4. Install the dependencies:
```
pip install -r requirements.txt
pip install -r examples/requirements.txt
```
5. Make the necessary directories observable by Python (`<SOLVER_SELECTOR_DIRECTORY>` and `<POREPY_DIRECTORY>` are the directories of the solver selector and PorePy, respectively):
```
export PYTHONPATH=$PYTHONPATH:<SOLVER_SELECTOR_DIRECTORY>/src
export PYTHONPATH=$PYTHONPATH:<SOLVER_SELECTOR_DIRECTORY>/examples
export PYTHONPATH=$PYTHONPATH:<POREPY_DIRECTORY>/examples
```
6. Run `pytest` to verify the installation:
```
pytest <SOLVER_SELECTOR_DIRECTORY>
```

## Reproducing the experiments
The code to reproduce the experiments is in the [examples](./examples/) directory. To run them all, use:
```
cd examples
bash run.bash <NUM_REPEATS>
```
where `<NUM_REPEATS>` is an integer representing how many times each experiment should be repeated. Alternatively, use `run.bash` scripts at the subdirectories of the experiments. The results are visualized in the `results_*.ipynb` notebooks. The Jupyter Notebooks server is one of the dependencies of the experiments, so it should be accessible by the command:
```
jupyter notebook
```

## TODO:
* License
* Docker image
* Reference to the paper
* Citing
