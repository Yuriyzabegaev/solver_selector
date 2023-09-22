from data_scripts import append_experiment_name, get_newest_data_paths
from thermal_model import make_thermal_setup
from thermal_solvers import make_thermal_solver_space

from solver_selector.simulation_runner import make_simulation_runner

experiment_path = append_experiment_name(__file__)
data_s = get_newest_data_paths("../1/thermal_dynamic", n_newest=1)
data_m = get_newest_data_paths("thermal_coldstart_m", n_newest=1)
assert len(data_s) == 1
assert len(data_m) == 1

experiment_path = append_experiment_name(__file__)
print("Starting experiment:", experiment_path.name)
simulation = make_thermal_setup(model_size="large")
solver_space = make_thermal_solver_space(solver="dynamic")
simulation_runner = make_simulation_runner(
    solver_space=solver_space,
    params={
        "save_statistics_path": experiment_path,
        "print_solver": True,
        "save_statistics_path": experiment_path,
        "print_solver": True,
        "regressor": "stacking",
        "stacking_datasets": [data_s, data_m],
    },
)

simulation_runner.run_simulation(simulation)
print("Done, saved at:", experiment_path)
