from data_scripts import append_experiment_name
from mandel_model import make_mandel_setup
from mandel_solvers import make_mandel_solver_space

from solver_selector.simulation_runner import make_simulation_runner

experiment_path = append_experiment_name(__file__)
print("Starting experiment:", experiment_path.name)
solver_space = make_mandel_solver_space(l_factor="dynamic")
simulation = make_mandel_setup(model_size="small")
simulation_runner = make_simulation_runner(
    solver_space=solver_space,
    params={"save_statistics_path": experiment_path, "print_solver": True},
)

simulation_runner.run_simulation(simulation)
print("Done, saved at:", experiment_path)
