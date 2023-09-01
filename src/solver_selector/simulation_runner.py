from solver_selector.solver_space import SolverConfigNode
from solver_selector.solver_selector import SolverSelector, RewardPicker
from solver_selector.utils import TimerContext
from solver_selector.performance_predictor import PerformancePredictorEpsGreedy
from solver_selector.data_structures import (
    ProblemContext,
    NonlinearSolverStats,
    SolverSelectionData,
)
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class Solver(ABC):
    """A solver to make a time step"""

    @abstractmethod
    def make_time_step(self, simulation: "SimulationModel") -> NonlinearSolverStats:
        """Perform an attempt to make a time step. Successfully or not, it returns
        performance data."""


class SimulationModel(ABC):
    """A class for the simulation-software-independent solver selector to iteract
    with the specific simulation software. What it does is:
    - assembles a solver from a configuration.
    - provides required data to the solver, e.g. degrees of freedom.
    - characterizes the current state of the simulation by some context.
    - tells when the simulation is done.

    """

    @abstractmethod
    def get_context(self) -> ProblemContext:
        """Get current simulation context that characterizes current simulation state."""

    @abstractmethod
    def assemble_solver(self, solver_config: dict) -> Solver:
        """Build a solver algorithm that implements the provided configuration."""

    def before_time_step(self):
        """Called before starting the new time step.

        Note: It is called before evaluating simulation characteristics with
            `get_context`. The work of updating the simulation characteristics should be
            done here.

        """

    def after_time_step(self, solver_selection_data: SolverSelectionData):
        """Called after finishing the time step."""

    def after_simulation(self):
        """Called once when the simulation successfuly finishes."""

    @abstractmethod
    def is_complete(self) -> bool:
        """Whether the simulation is over."""


class SimulationRunner:
    def __init__(
        self,
        solver_selector: SolverSelector,
        reward_picker: RewardPicker,
        params: dict = None,
    ):
        save_statistics_path = params.get("save_statistics_path")
        if save_statistics_path is not None:
            save_statistics_path = str(save_statistics_path)
        self.save_statistics_path: str | None = save_statistics_path
        params = params or {}
        self.params = {"print_solver": False, "print_time": False} | params
        self.solver_selector: SolverSelector = solver_selector
        self.reward_picker: RewardPicker = reward_picker
        self.update_selector_times: list[float] = []
        self.select_solver_times: list[float] = []

    def run_simulation(self, simulation: SimulationModel):
        """The main simulation loop."""
        solver_selector = self.solver_selector
        solver_space = solver_selector.solver_space
        while not simulation.is_complete():
            # Inform the simulation that the new time step starts.
            simulation.before_time_step()

            with TimerContext() as timer_select_solver:
                # Evaluating simulation characteristics on the current time step.
                context = simulation.get_context()

                # Selecting a solver and making its config.
                predicted_solver = solver_selector.select_solver(context)
                solver_config = solver_space.config_from_decision(
                    predicted_solver.decision
                )
                if self.params["print_solver"]:
                    solver_config_for_print = solver_space.config_from_decision(
                        predicted_solver.decision, optimized_only=True
                    )
                    print(solver_space.format_config(solver_config_for_print))

                # Assembling the solver based on the config.
                solver = simulation.assemble_solver(solver_config)

            # Actually solving the time step system. The time step can be successful, or
            # the solver may fail.
            with TimerContext() as timer_solve:
                performance_data = solver.make_time_step(simulation)

            with TimerContext() as timer_update_selector:
                # Grading the performance with the provided reward function.
                rewards = self.reward_picker.pick_rewards(performance_data)
                solver_selection_data = SolverSelectionData(
                    nonlinear_solver_stats=performance_data,
                    prediction=predicted_solver,
                    rewards=rewards,
                    work_time=timer_solve.elapsed_time,
                )
                # Updating the solver selector with newly obtainer performance data.
                solver_selector.learn_performance_online(solver_selection_data)

            self.select_solver_times.append(timer_select_solver.elapsed_time)
            self.update_selector_times.append(timer_update_selector.elapsed_time)

            simulation.after_time_step(solver_selection_data)

            if self.params["print_time"]:
                print(f"Select solver time: {timer_select_solver.elapsed_time:2e}")
                print(f"Solve time: {timer_solve.elapsed_time:2e}")
                print(f"Update selector time: {timer_update_selector.elapsed_time:2e}")

            if self.save_statistics_path is not None:
                np.save(self.save_statistics_path, self.solver_selector.memory)

        simulation.after_simulation()
        print("Simulation finished successfully.")


def make_simulation_runner(
    solver_space: SolverConfigNode, params: dict = None
) -> SimulationRunner:
    """Convenience function to initialize all the solver selection objects."""
    params = params or {}
    reward_picker = RewardPicker()

    all_solvers = solver_space.get_all_solvers()
    print(f"Selecting from {len(all_solvers)} solvers.")
    for i, solver_template in enumerate(all_solvers):
        default = solver_template.use_defaults()
        conf = solver_space.config_from_decision(decision=default, optimized_only=True)
        print(i, solver_space.format_config(conf))

    predictor = params.get("predictir", "eps_greedy")
    if predictor == "eps_greedy":
        print("Using epsilon-greedy exploration")
        exploration = params.get("exploration", 0.5)
        exploration_rate = params.get("exploration_rate", 0.9)
        samples_before_fit = params.get("samples_before_fit", 10)

        predictors = []
        for solver_template in all_solvers:
            predictors.append(
                PerformancePredictorEpsGreedy(
                    decision_template=solver_template,
                    exploration=exploration,
                    exploration_rate=exploration_rate,
                    samples_before_fit=samples_before_fit,
                )
            )
    solver_selector = SolverSelector(
        solver_space=solver_space,
        predictors=predictors,
    )
    return SimulationRunner(
        solver_selector=solver_selector, reward_picker=reward_picker, params=params
    )
