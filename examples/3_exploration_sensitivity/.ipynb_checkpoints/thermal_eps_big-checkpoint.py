import sys
import numpy as np
from data_scripts import append_experiment_name
from thermal_model import make_thermal_setup
from thermal_solvers import make_thermal_solver_space

from solver_selector.simulation_runner import make_simulation_runner

np.random.seed(int(sys.argv[1]))

experiment_path = append_experiment_name(__file__)
print("Starting experiment:", experiment_path.name)
solver_space = make_thermal_solver_space(solver="dynamic")
simulation = make_thermal_setup(model_size="medium")
simulation_runner = make_simulation_runner(
    solver_space=solver_space,
    params={
        "save_statistics_path": experiment_path,
        "print_solver": True,
        "exploration": 0.7,
    },
)

simulation_runner.run_simulation(simulation)
print("Done, saved at:", experiment_path)
