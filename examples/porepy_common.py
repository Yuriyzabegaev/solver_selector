from solvers_common import LinearSolver, LinearSolverStatistics
from porepy import SolutionStrategy
from solver_selector.utils import TimerContext
from solver_selector.data_structures import (
    NonlinearIterationStats,
    NonlinearSolverStats,
)
from solver_selector.simulation_runner import Solver, SimulationModel
import numpy as np
from typing import TYPE_CHECKING


class PorepySimulation(SimulationModel):
    porepy_setup: SolutionStrategy


class LinearSolverFailed(Exception):
    pass


class PorepyNewtonSolver(Solver):
    def __init__(self, linear_solver: LinearSolver) -> None:
        self.linear_solver: LinearSolver = linear_solver
        self.params = {"max_iterations": 15, "nl_convergence_tol": 1e-6}
        self.error_norms = []
        self.residual_norms = []

    def make_time_step(self, simulation: PorepySimulation) -> NonlinearSolverStats:
        return self.solve(simulation.porepy_setup)

    def newton_iteration(
        self,
        model: SolutionStrategy,
        init_sol: np.ndarray,
        prev_sol: np.ndarray,
        iteration_counter: int,
    ) -> dict:
        try:
            # Assemble the new Jacobian matrix
            with TimerContext() as assembly_timer:
                # Re-discretize the nonlinear term
                model.before_nonlinear_iteration()
                model.assemble_linear_system()
                mat, rhs = model.linear_system
                # If the matrix is very bad, the linear solver will anyway fail.
                if np.any(np.isnan(mat.data)):
                    raise LinearSolverFailed("Jacobian contains nans.")

            #  Update the preconditioner with the new matrix
            with TimerContext() as update_timer:
                self.linear_solver.update(mat)

            # Solve the Jacobian linear system
            with TimerContext() as solve_timer:
                sol, linear_stats = self.linear_solver.solve(rhs)

            # Check convergence
            with TimerContext() as convergence_timer:
                model.after_nonlinear_iteration(sol)

                if model._is_nonlinear_problem():
                    error_norm, is_converged, is_diverged = model.check_convergence(
                        sol, prev_sol, init_sol, self.params
                    )
                else:
                    if linear_stats.is_converged:
                        is_diverged = False
                        is_converged = True
                        error_norm = linear_stats.residual_decrease
                    else:
                        is_diverged = True
                        is_converged = False
                        error_norm = -1

                if linear_stats.residual_decrease is not None:
                    self.residual_norms.append(linear_stats.residual_decrease)
                    if linear_stats.residual_decrease > 1e16:
                        is_diverged = True

                if not linear_stats.is_converged:
                    print("Linear solver failed.")
                if linear_stats.is_diverged:
                    print("Linear solver diverged.")
                if is_diverged:
                    print("Nonlinear solver diverged.")

                self.error_norms.append(error_norm)
                print(
                    f"Newton iter: {iteration_counter}, error: {self.error_norms[-1]}, "
                    f"linear iters: {linear_stats.num_iters}"
                )
        except LinearSolverFailed as e:
            print(e)
            is_diverged = True
            is_converged = False
            linear_stats = LinearSolverStatistics(
                num_iters=-1, is_converged=False, is_diverged=True
            )
        return {
            "sol": sol,
            "is_converged": is_converged,
            "is_diverged": is_diverged,
            "linear_stats": linear_stats,
            "solve_time": solve_timer.elapsed_time,
            "assembly_time": assembly_timer.elapsed_time,
            "update_time": update_timer.elapsed_time,
        }

    def solve(self, porepy_setup: SolutionStrategy) -> NonlinearSolverStats:
        model = porepy_setup

        sol = model.equation_system.get_variable_values(time_step_index=0)
        init_sol = sol
        self.error_norms = []
        self.residual_norms = []
        iteration_stats = []

        iteration_counter = 0
        for iteration_counter in range(self.params["max_iterations"]):
            with TimerContext() as work_timer:
                iteration = self.newton_iteration(
                    model, init_sol, sol, iteration_counter
                )
            is_converged = iteration["is_converged"]
            is_diverged = iteration["is_diverged"]
            sol = iteration["sol"]
            linear_stats: LinearSolverStatistics = iteration["linear_stats"]
            if (res_decrease := linear_stats.residual_decrease) is None:
                res_decrease = -1
            iteration_stats.append(
                NonlinearIterationStats(
                    work_time=work_timer.elapsed_time,
                    solve_linear_system_time=iteration["solve_time"],
                    assembly_time=iteration["assembly_time"],
                    update_preconditioner_time=iteration["update_time"],
                    linear_solver_converged=linear_stats.is_converged,
                    num_linear_iterations=linear_stats.num_iters,
                    linear_residual_decrease=res_decrease,
                )
            )

            if is_converged or is_diverged:
                break

        if len(self.residual_norms) > 1:
            print("||F||/||F_0||:", self.residual_norms[-1] / self.residual_norms[0])

        if is_converged:
            model.after_nonlinear_convergence(sol, self.error_norms, iteration_counter)
        if not is_converged:
            try:
                model.after_nonlinear_failure(sol, self.error_norms, iteration_counter)
            except ValueError:
                pass

        return NonlinearSolverStats(
            is_converged=is_converged,
            is_diverged=is_diverged,
            num_nonlinear_iterations=len(iteration_stats),
            nonlinear_error=self.error_norms,
            iterations=iteration_stats,
        )
