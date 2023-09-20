from pytest import raises

from tests_common import DummpyProblemContext

from solver_selector.data_structures import (
    NonlinearIterationStats,
    NonlinearSolverStats,
)
from solver_selector.simulation_runner import (
    SimulationModel,
    Solver,
    make_simulation_runner,
)
from solver_selector.solver_space import ConstantNode, KrylovSolverDecisionNode


def generate_solver_stats(multiplier, is_success=True):
    return NonlinearSolverStats(
        is_converged=is_success,
        is_diverged=not is_success,
        nonlinear_error=[0, 0],
        iterations=[
            NonlinearIterationStats(
                work_time=3 * i * multiplier,
                solve_linear_system_time=i * multiplier,
                assembly_time=i * multiplier,
                update_preconditioner_time=i * multiplier,
                linear_solver_converged=True,
                num_linear_iterations=i * multiplier,
                linear_residual_decrease=1e-8,
            )
            for i in range(1, 4)
        ],
    )


class DummySolver1(Solver):
    """This one is faster"""

    def make_time_step(self, simulation: SimulationModel) -> NonlinearSolverStats:
        assert isinstance(simulation, DummySimulation)
        simulation.time_step_idx += 1
        return generate_solver_stats(multiplier=1)


class DummySolver2(Solver):
    """This one is slower"""

    def make_time_step(self, simulation: SimulationModel) -> NonlinearSolverStats:
        assert isinstance(simulation, DummySimulation)
        simulation.time_step_idx += 1
        return generate_solver_stats(multiplier=10)


class DummySolver3(Solver):
    """This one always fails"""

    def make_time_step(self, simulation: SimulationModel) -> NonlinearSolverStats:
        assert isinstance(simulation, DummySimulation)
        simulation.time_step_idx += 1
        return generate_solver_stats(multiplier=100, is_success=False)


class DummySimulation(SimulationModel):
    def __init__(self) -> None:
        self.time_step_idx = 0

    def get_context(self):
        return DummpyProblemContext()

    def assemble_solver(self, solver_config: dict) -> Solver:
        solver_name = list(solver_config.keys())[0]
        if solver_name == "solver1":
            return DummySolver1()
        elif solver_name == "solver2":
            return DummySolver2()
        elif solver_name == "solver3":
            return DummySolver3()
        raise ValueError

    def is_complete(self) -> bool:
        return self.time_step_idx >= 10


def test_simulation_runner():
    solver_space = KrylovSolverDecisionNode(
        [
            ConstantNode("solver1"),
            ConstantNode("solver2"),
        ]
    )
    simulation_runner = make_simulation_runner(
        solver_space, {"print_solver": True, "print_time": True}
    )

    simulation = DummySimulation()

    simulation_runner.run_simulation(simulation)

    assert simulation.time_step_idx == 10


def test_simulation_runner_bad_solver():
    solver_space = KrylovSolverDecisionNode([ConstantNode("solver3")])
    simulation_runner = make_simulation_runner(
        solver_space, {"print_solver": True, "print_time": True}
    )

    simulation = DummySimulation()

    with raises(ValueError):
        simulation_runner.run_simulation(simulation)


if __name__ == "__main__":
    test_simulation_runner()
    test_simulation_runner_bad_solver()