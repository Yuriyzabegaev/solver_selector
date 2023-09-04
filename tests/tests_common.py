from typing import Sequence
import numpy as np
from solver_selector.data_structures import (
    ProblemContext,
    SolverSelectionData,
    NonlinearIterationStats,
    NonlinearSolverStats,
    PerformancePredictionData,
)
from solver_selector.data_structures import ProblemContext


class DummpyProblemContext(ProblemContext):
    def get_array(self) -> np.ndarray:
        return np.array([1, 2, 3], dtype=float)


def generate_synthetic_data(
    prediction: PerformancePredictionData, seed: int
) -> Sequence[SolverSelectionData]:
    return [
        SolverSelectionData(
            work_time=num_linear_systems * 10,
            nonlinear_solver_stats=NonlinearSolverStats(
                is_converged=True,
                is_diverged=False,
                num_nonlinear_iterations=num_linear_systems,
                nonlinear_error=[0, 0],
                iterations=[
                    NonlinearIterationStats(
                        work_time=i + seed,
                        solve_linear_system_time=i + seed,
                        assembly_time=i + seed,
                        update_preconditioner_time=i + seed,
                        linear_solver_converged=True,
                        num_linear_iterations=i + seed,
                        linear_residual_decrease=1e-8,
                    )
                    for i in range(1, num_linear_systems + 1)
                ],
            ),
            prediction=prediction,
            rewards=[i * 2 + seed for i in range(1, num_linear_systems + 1)],
        )
        for num_linear_systems in [4, 6]
    ]
