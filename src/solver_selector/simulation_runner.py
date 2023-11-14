from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from solver_selector.data_structures import (
    NonlinearSolverStats,
    ProblemContext,
    SolverSelectionData,
)
from solver_selector.solver_selector import (
    RewardPicker,
    SolverSelector,
    make_solver_selector,
)
from solver_selector.solver_space import SolverConfigNode
from solver_selector.utils import TimerContext


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

    def after_time_step_success(self, solver_selection_data: SolverSelectionData):
        """Called after successfully finishing the time step."""

    def after_time_step_failure(self, solver_selection_data: SolverSelectionData):
        """Called after unsuccessfully finishing the time step. By default, raises an
        error.

        """
        raise ValueError("Simulation time step is not done.")

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
        params: Optional[dict] = None,
    ):
        params = params or {}
        save_statistics_path = params.get("save_statistics_path")
        if save_statistics_path is not None:
            save_statistics_path = str(save_statistics_path)
        self.save_statistics_path: str | None = save_statistics_path
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
                    config=solver_config.copy(),
                    rewards=rewards,
                    work_time=timer_solve.elapsed_time,
                )
                # Updating the solver selector with newly obtainer performance data.
                solver_selector.learn_performance_online(solver_selection_data)

            self.select_solver_times.append(timer_select_solver.elapsed_time)
            self.update_selector_times.append(timer_update_selector.elapsed_time)

            if performance_data.is_converged:
                simulation.after_time_step_success(solver_selection_data)
            else:
                simulation.after_time_step_failure(solver_selection_data)

            if self.params["print_time"]:
                print(f"Select solver time: {timer_select_solver.elapsed_time:2e}")
                print(f"Solve time: {timer_solve.elapsed_time:2e}")
                print(f"Update selector time: {timer_update_selector.elapsed_time:2e}")

            if self.save_statistics_path is not None:
                np.save(
                    self.save_statistics_path,
                    self.solver_selector.memory,  # type: ignore[arg-type]
                )

        if self.save_statistics_path is not None:
            metrics_path = (
                f"{self.save_statistics_path.removesuffix('.npy')}_metrics.npy"
            )
            np.save(
                metrics_path,
                {
                    "select_solver_times": self.select_solver_times,
                    "update_selector_times": self.update_selector_times,
                },  # type:ignore[arg-type]
            )
        simulation.after_simulation()
        print("Simulation finished successfully.")


def make_simulation_runner(
    solver_space: SolverConfigNode, params: Optional[dict] = None
) -> SimulationRunner:
    """Convenience function to initialize all the solver selection objects."""
    params = params or {}
    reward_picker = RewardPicker()

    solver_selector = make_solver_selector(solver_space=solver_space, params=params)

    return SimulationRunner(
        solver_selector=solver_selector, reward_picker=reward_picker, params=params
    )
