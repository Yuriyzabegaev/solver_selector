from solver_selector.data_structures import (
    NonlinearSolverStats,
    NonlinearIterationStats,
)
from solver_selector.simulation_runner import (
    SimulationModel,
    Solver,
    make_simulation_runner,
)
from solver_selector.solver_space import (
    ConstantNode,
    KrylovSolverDecisionNode,
    ForkNodeNames,
)
from tests_common import DummpyProblemContext


def generate_solver_stats(multiplier):
    return NonlinearSolverStats(
        is_converged=True,
        is_diverged=False,
        raised_error=False,
        num_nonlinear_iterations=1,
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


class DummySimulation(SimulationModel):
    def __init__(self) -> None:
        self.time_step_idx = 0

    def get_context(self):
        return DummpyProblemContext()

    def assemble_solver(self, solver_config: dict) -> Solver:
        value = solver_config[ForkNodeNames.krylov_solver_picker]
        if value == "solver1":
            return DummySolver1()
        elif value == "solver2":
            return DummySolver2()
        raise ValueError

    def before_time_step(self):
        pass

    def is_complete(self) -> bool:
        return self.time_step_idx >= 10


def test_simulation_runner():
    solver_space = KrylovSolverDecisionNode(
        [
            ConstantNode("solver1"),
            ConstantNode("solver2"),
        ]
    )
    simulation_runner = make_simulation_runner(solver_space)

    simulation = DummySimulation()

    simulation_runner.run_simulation(simulation)

    assert simulation.time_step_idx == 10


if __name__ == "__main__":
    test_simulation_runner()
