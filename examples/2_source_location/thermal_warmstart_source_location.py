from data_scripts import append_experiment_name, get_newest_data_paths
from thermal_model import make_thermal_setup
from thermal_solvers import make_thermal_solver_space

from solver_selector.simulation_runner import make_simulation_runner

experiment_path = append_experiment_name(__file__)
load_data_paths = get_newest_data_paths("../1/thermal_dynamic", n_newest=1)
# load_data_paths += get_newest_data_paths("thermal_coldstart_m", n_newest=1)
assert len(load_data_paths) == 1

experiment_path = append_experiment_name(__file__)
print("Starting experiment:", experiment_path.name)
simulation = make_thermal_setup(model_size="small", source_location=1)
solver_space = make_thermal_solver_space(solver="dynamic")
simulation_runner = make_simulation_runner(
    solver_space=solver_space,
    params={
        "save_statistics_path": experiment_path,
        "print_solver": True,
        "load_statistics_paths": load_data_paths,
        # "exploration": 0,
    },
)

simulation_runner.run_simulation(simulation)
print("Done, saved at:", experiment_path)
