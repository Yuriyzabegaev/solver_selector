import numpy as np
from porepy import SolutionStrategy
from solvers_common import LinearSolver, LinearSolverStatistics

from solver_selector.data_structures import (
    NonlinearIterationStats,
    NonlinearSolverStats,
    SolverSelectionData,
)
from solver_selector.simulation_runner import SimulationModel, Solver
from solver_selector.utils import TimerContext


class PorepySimulation(SimulationModel):
    porepy_setup: SolutionStrategy

    def _compute_cfl(self):
        setup = self.porepy_setup
        subdomains = setup.mdg.subdomains()
        velosity = setup.darcy_flux(subdomains) / setup.fluid_viscosity(subdomains)
        velosity = velosity.evaluate(setup.equation_system).val
        length = setup.mdg.subdomains()[0].face_areas
        time_step = setup.time_manager.dt
        CFL = velosity * time_step / length
        CFL_max = abs(CFL).max()
        CFL_mean = abs(CFL).mean()
        return CFL_max, CFL_mean

    def is_complete(self) -> bool:
        time_manager = self.porepy_setup.time_manager
        return time_manager.time >= time_manager.time_final

    def after_time_step_success(self, solver_selection_data: SolverSelectionData):
        model = self.porepy_setup
        model.time_manager.increase_time()
        model.time_manager.increase_time_index()

    def after_simulation(self):
        self.porepy_setup.after_simulation()

    def before_time_step(self) -> None:
        model = self.porepy_setup
        print(
            "\nTime step {} at time {:.1e} of {:.1e} with time step {:.1e}".format(
                model.time_manager.time_index,
                model.time_manager.time,
                model.time_manager.time_final,
                model.time_manager.dt,
            )
        )
        model.before_nonlinear_loop()


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
            model.after_nonlinear_iteration(sol)

            if model._is_nonlinear_problem():
                error_norm, is_converged, is_diverged = model.check_convergence(
                    sol, prev_sol, init_sol, self.params
                )
                # if not linear_stats.is_converged and not is_converged:
                #     is_diverged = True
                #     # This is kind of strict, but currently we do not consider
                #     # inexact Newton.

            else:  # Linear problem
                if linear_stats.is_converged:
                    is_diverged = False
                    is_converged = True
                    error_norm = linear_stats.residual_decrease
                else:
                    is_diverged = True
                    is_converged = False
                    error_norm = -1

            self.residual_norms.append(np.linalg.norm(rhs))
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
            sol = prev_sol
            is_diverged = True
            is_converged = False
            linear_stats = LinearSolverStatistics(
                num_iters=-1, is_converged=False, is_diverged=True
            )
            solve_time = 0
            update_time = 0
            assembly_time = assembly_timer.elapsed_time
        else:
            solve_time = solve_timer.elapsed_time
            assembly_time = assembly_timer.elapsed_time
            update_time = update_timer.elapsed_time
        return {
            "sol": sol,
            "is_converged": is_converged,
            "is_diverged": is_diverged,
            "linear_stats": linear_stats,
            "solve_time": solve_time,
            "assembly_time": assembly_time,
            "update_time": update_time,
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
            # We do not compute residual norm on the last iteration, so it must be less
            # than on the one second from last iteration.
            print(
                "||F||/||F_0|| < "
                f"{self.residual_norms[-1] / self.residual_norms[0]:.2e}"
            )

        if is_converged:
            model.after_nonlinear_convergence(sol, self.error_norms, iteration_counter)
        else:
            try:
                model.after_nonlinear_failure(sol, self.error_norms, iteration_counter)
            except ValueError:
                pass

        return NonlinearSolverStats(
            is_converged=is_converged,
            is_diverged=is_diverged,
            nonlinear_error=self.error_norms,
            iterations=iteration_stats,
        )
