from pathlib import Path

from data_scripts import append_experiment_name, get_newest_data_paths
from mandel_model import make_mandel_setup
from mandel_solvers import make_mandel_solver_space

from solver_selector.simulation_runner import make_simulation_runner

path = Path(__file__).parent / "../2"
load_data_paths = get_newest_data_paths(path / "poro_coldstart_s", n_newest=1)
load_data_paths += get_newest_data_paths(path / "poro_coldstart_m", n_newest=1)
assert len(load_data_paths) == 2

experiment_path = append_experiment_name(__file__)
print("Starting experiment:", experiment_path.name)
solver_space = make_mandel_solver_space(l_factor="dynamic")
simulation = make_mandel_setup(model_size="large")
simulation_runner = make_simulation_runner(
    solver_space=solver_space,
    params={
        "save_statistics_path": experiment_path,
        "load_statistics_paths": load_data_paths,
        "print_solver": True,
        "exploration": 0,
    },
)

simulation_runner.run_simulation(simulation)
print("Done, saved at:", experiment_path)
