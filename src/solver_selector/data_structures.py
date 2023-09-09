from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from solver_selector.solver_space import Decision


@dataclass(kw_only=True, slots=True, frozen=True)
class NonlinearIterationStats:
    """Data for one linear system / one Newton iteration."""

    work_time: float
    solve_linear_system_time: float
    assembly_time: float
    update_preconditioner_time: float
    linear_solver_converged: bool
    num_linear_iterations: int
    linear_residual_decrease: float


@dataclass(kw_only=True, slots=True, frozen=True)
class NonlinearSolverStats:
    """Data for one simulation time step. Includes one or several linear systems."""

    is_converged: bool
    is_diverged: bool
    # num_nonlinear_iterations: int  # about to remove
    nonlinear_error: Sequence[float]
    iterations: Sequence[NonlinearIterationStats]


@dataclass(frozen=True, kw_only=True, slots=True)
class ProblemContext:
    """Data that characterizes the simulation state at the given time step."""

    def get_array(self) -> np.ndarray:
        raise NotImplementedError

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(kw_only=True, slots=True, frozen=True)
class PerformancePredictionData:
    """Represents the decision made for the given context."""

    score: float
    decision: Decision
    context: ProblemContext


@dataclass(kw_only=True, slots=True, frozen=True)
class SolverSelectionData:
    """The full data related to the prosses of solver selection. Encloses all the other
    data structures related to the specific details of solver selection. Used to store
    and reutilize the data in future solver selection runs for the warm start.

    """

    nonlinear_solver_stats: NonlinearSolverStats
    prediction: PerformancePredictionData
    config: dict
    rewards: Sequence[float]
    work_time: float
