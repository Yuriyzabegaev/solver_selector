from solver_selector.solver_space import (
    KrylovSolverNode,
    SolverConfigNode,
    ParametersNode,
    ConstantNode,
    NumericalParameter,
    DecisionNodeNames,
)
from solver_selector.simulation_runner import SimulationModel
from typing_extensions import Self
import numpy as np
from scipy.sparse.linalg import LinearOperator, spsolve, gmres
from scipy.sparse import csr_matrix
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import petsc4py

petsc4py.init()
from petsc4py import PETSc


#  ------------------- Solver space -------------------


class LinearSolverNames:
    richardson = "richardson"
    gmres = "gmres"
    none = "none"
    direct = "direct"


class PreconditionerNames:
    ilu = "ilu"
    amg = "amg"
    none = "none"


class GMRESNode(KrylovSolverNode):
    type = LinearSolverNames.gmres


class NodePreconditionerILU(SolverConfigNode):
    type = PreconditionerNames.ilu

    def __init__(
        self, name: str = None, drop_tol: float | NumericalParameter = 1e-8
    ) -> None:
        self.drop_tol = drop_tol
        self.params = ParametersNode(
            name="ilu_params", continuous_actions={"drop_tol": drop_tol}
        )
        super().__init__(
            children=[self.params],
            name=name or self.type,
        )

    def copy(self) -> Self:
        return type(self)(name=self.name, drop_tol=self.drop_tol)


class NodePreconditionerAMG(ConstantNode):
    type = PreconditionerNames.amg


class DirectSolverNode(SolverConfigNode):
    type = LinearSolverNames.direct


class NoKrylovSolverNode(KrylovSolverNode):
    """Placeholder to apply a preconditioner only."""

    type = LinearSolverNames.none


#  ------------------- Solvers implementations -------------------


@dataclass(kw_only=True, slots=True, frozen=True)
class LinearSolverStatistics:
    num_iters: int
    is_converged: bool
    is_diverged: bool
    residual_decrease: Optional[float] = None


class Preconditioner(ABC):
    def __init__(self, config: dict = None) -> None:
        self.config: dict = config or {}
        self.mat: csr_matrix = None

    def update(self, mat: csr_matrix) -> None:
        self.mat = mat

    @abstractmethod
    def apply(self, b: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def aslinearoperator(self) -> LinearOperator:
        return LinearOperator(
            matvec=self.apply, shape=self.mat.shape, dtype=self.mat.dtype
        )


class LinearSolver:
    def __init__(
        self,
        preconditioner: Preconditioner,
        config: dict = None,
    ) -> None:
        self.config: dict = config or {}
        self.mat: csr_matrix = None
        self.preconditioner: Preconditioner = preconditioner

    def update(self, mat: csr_matrix) -> None:
        self.mat = mat
        self.preconditioner.update(mat)

    @abstractmethod
    def solve(
        self, b: np.ndarray, x0: np.ndarray = None, tol: float = None
    ) -> tuple[np.ndarray, LinearSolverStatistics]:
        """Returns the solution and statistics."""
        raise NotImplementedError


class SplittingSchur(Preconditioner):
    def __init__(
        self,
        linear_solver_primary: LinearSolver,
        linear_solver_secondary: LinearSolver,
        config: dict,
    ):
        assert "method" in config
        self.linear_solver_primary: LinearSolver = linear_solver_primary
        self.linear_solver_secondary: LinearSolver = linear_solver_secondary

        # They must be initialized in a problem-specific subclass
        self.primary_ind: np.ndarray
        self.secondary_ind: np.ndarray
        self.mat_01: csr_matrix
        self.mat_00: csr_matrix
        self.mat_10: csr_matrix
        super().__init__(config=config)

    def apply(self, b):
        rhs_0 = b[self.primary_ind]
        rhs_1 = b[self.secondary_ind]

        method = self.config["method"]
        if method == "upper":
            sol_1, _ = self.linear_solver_secondary.solve(rhs_1)
            sol_0, _ = self.linear_solver_primary.solve(rhs_0 - self.mat_01 @ sol_1)
        elif method == "lower":
            sol_0, _ = self.linear_solver_primary.solve(rhs_0)
            sol_1, _ = self.linear_solver_secondary.solve(rhs_1 - self.mat_10 @ sol_0)
        elif method == "full":
            tmp_sol, _ = self.linear_solver_primary.solve(rhs_0)
            sol_1, _ = self.linear_solver_secondary.solve(rhs_1 - self.mat_10 @ tmp_sol)
            sol_0, _ = self.linear_solver_primary.solve(rhs_0 - self.mat_01 @ sol_1)
        else:
            raise NotImplementedError(method)

        result = np.zeros_like(b)
        result[self.primary_ind] = sol_0
        result[self.secondary_ind] = sol_1
        return result

    def residual_secondary(self, sol, rhs):
        return np.linalg.norm(
            self.linear_solver_secondary.mat.dot(sol) - rhs
        ) / np.linalg.norm(rhs)

    def residual_primary(self, sol, rhs):
        return np.linalg.norm(
            self.linear_solver_primary.mat.dot(sol) - rhs
        ) / np.linalg.norm(rhs)


class PreconditionerPetscAMG(Preconditioner):
    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        self.pc = PETSc.PC().create()
        options = PETSc.Options()
        # options["pc_type"] = "gamg"
        # options['pc_gamg_agg_nsmooths'] = 1
        # options["mg_levels_ksp_type"] = "chebyshev"
        # options["mg_levels_pc_type"] = "jacobi"
        # options["mg_levels_ksp_chebyshev_esteig_steps"] = 10

        options["pc_type"] = "hypre"
        options["pc_hypre_type"] = "boomeramg"
        options["pc_hypre_boomeramg_max_iter"] = config.get("max_iter", 1)
        options["pc_hypre_boomeramg_cycle_type"] = config.get("cycle", "V")
        # options.setValue('pc_hypre_boomeramg_relax_type_all', 'Chebyshev')
        # options.setValue('pc_hypre_boomeramg_smooth_type', 'Pilut')

        self.pc.setFromOptions()
        self.petsc_A = PETSc.Mat()

    def update(self, mat):
        super().update(mat)
        self.petsc_A.createAIJ(size=mat.shape, csr=(mat.indptr, mat.indices, mat.data))
        self.pc.setOperators(self.petsc_A)
        self.pc.setUp()

    def __del__(self):
        self.pc.destroy()
        self.petsc_A.destroy()

    def apply(self, b):
        petsc_b = PETSc.Vec().createWithArray(b)
        x = self.petsc_A.createVecLeft()
        x.set(0.0)
        self.pc.apply(petsc_b, x)
        res = x.getArray()
        petsc_b.destroy()
        x.destroy()
        return res


class PreconditionerPetscILU(Preconditioner):
    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        self.pc = PETSc.PC().create()
        options = PETSc.Options()
        options.setValue("pc_type", "ilu")
        options.setValue("pc_factor_levels", 0)
        options.setValue("pc_factor_diagonal_fill", None)  # Doesn't affect
        options.setValue("pc_factor_nonzeros_along_diagonal", None)
        self.pc.setFromOptions()

        self.petsc_A = PETSc.Mat()

    def update(self, mat):
        super().update(mat)
        self.petsc_A.createAIJ(size=mat.shape, csr=(mat.indptr, mat.indices, mat.data))
        self.pc.setOperators(self.petsc_A)
        self.pc.setUp()

    def __del__(self):
        self.pc.destroy()
        self.petsc_A.destroy()

    def apply(self, b):
        petsc_b = PETSc.Vec().createWithArray(b)
        x = self.petsc_A.createVecLeft()
        x.set(0.0)
        self.pc.apply(petsc_b, x)
        res = x.getArray()
        petsc_b.destroy()
        x.destroy()
        return res


class LinearSolverGMRES(LinearSolver):
    def __init__(
        self,
        preconditioner: Preconditioner,
        config: dict,
    ) -> None:
        # maxiter is a number of outer GMRES iterations.
        for param in ["restart", "tol", "maxiter"]:
            assert param in config
        super().__init__(preconditioner=preconditioner, config=config)

        self._num_iters: int = 0

    def gmres_callback(self, x):
        self._num_iters += 1

    def solve(self, b, x0=None, tol: float = None):
        self._num_iters = 0

        tol = tol or self.config["tol"]
        restart = self.config["restart"]
        maxiter = self.config["maxiter"]
        if x0 is None:
            res_0 = np.linalg.norm(b)
        else:
            res_0 = np.linalg.norm(self.mat.dot(x0) - b)
        x, info = gmres(
            A=self.mat,
            b=b,
            x0=x0,
            tol=tol,
            atol=0,
            restart=restart,
            maxiter=maxiter * restart,
            M=self.preconditioner.aslinearoperator(),
            callback=self.gmres_callback,
            callback_type="legacy",
        )

        res_end = np.linalg.norm(self.mat.dot(x) - b)

        return x, LinearSolverStatistics(
            num_iters=self._num_iters,
            is_converged=(info == 0),
            is_diverged=(info < 0),
            residual_decrease=float(res_end / res_0),
        )


class NoneLinearSolver(LinearSolver):
    def solve(self, b, x0=None, tol: float = None):
        return self.preconditioner.apply(b), LinearSolverStatistics(
            num_iters=1, is_converged=False, is_diverged=False
        )


class NonePreconditioner(Preconditioner):
    def apply(self, b: np.ndarray) -> np.ndarray:
        return b


class LinearSolverDirect(LinearSolver):
    def solve(self, b, x0=None, tol: float = None):
        return spsolve(self.mat, b), LinearSolverStatistics(
            num_iters=1, is_converged=True, is_diverged=False
        )


#  ------------------- Solver assembling routines -------------------


# def assemble_splitting(
#     self, splitting_config: dict, problem_description: ProblemDescription
# ):
#     splitting_type, splitting_options = list(splitting_config.items())[0]

#     linear_solver_primary = self.assemble_linear_solver(
#         linear_solver_config=splitting_options['primary'],
#         problem_description=problem_description,
#     )
#     linear_solver_secondary = self.assemble_linear_solver(
#         linear_solver_config=splitting_options['secondary'],
#         problem_description=problem_description,
#     )

#     if splitting_type == SplittingNames.schur:
#         return SplittingSchurDiag(
#             linear_solver_primary=linear_solver_primary,
#             linear_solver_secondary=linear_solver_secondary,
#             problem_description=problem_description,
#             config=splitting_options,
#         )
#     elif splitting_type == SplittingNames.fixed_stress:
#         return SplittingFixedStress(
#             linear_solver_primary=linear_solver_primary,
#             linear_solver_secondary=linear_solver_secondary,
#             problem_description=problem_description,
#             config=splitting_options,
#         )
#     elif splitting_type == SplittingNames.undrained_split:
#         return SplittingUndrained(
#             linear_solver_primary=linear_solver_primary,
#             linear_solver_secondary=linear_solver_secondary,
#             problem_description=problem_description,
#             config=splitting_options,
#         )
#     elif splitting_type == SplittingNames.schur_cd:
#         return SplittingSchurCD(
#             linear_solver_primary=linear_solver_primary,
#             linear_solver_secondary=linear_solver_secondary,
#             problem_description=problem_description,
#             config=splitting_options,
#         )
#     elif splitting_type == SplittingNames.cpr:
#         return SplittingCPR(
#             linear_solver_primary=linear_solver_primary,
#             linear_solver_secondary=linear_solver_secondary,
#             problem_description=problem_description,
#             config=splitting_options,
#         )

#     raise ValueError(splitting_type)


class SolverAssembler:
    def __init__(self, simulation: SimulationModel):
        self.simulation: SimulationModel = simulation

    def assemble_linear_solver(self, linear_solver_config: dict) -> LinearSolver:
        linear_solver_type, linear_solver_config = list(linear_solver_config.items())[0]

        preconditioner_config = linear_solver_config.get(
            DecisionNodeNames.preconditioner_picker, {}
        )
        parameters = linear_solver_config.get(DecisionNodeNames.parameters_picker, {})

        if len(preconditioner_config) == 0:
            preconditioner_config = {PreconditionerNames.none: {}}
        preconditioner = self.assemble_preconditioner(preconditioner_config)

        match linear_solver_type:
            case LinearSolverNames.none:
                solver = NoneLinearSolver(preconditioner, config=parameters)
            case LinearSolverNames.gmres:
                solver = LinearSolverGMRES(preconditioner, config=parameters)
            case LinearSolverNames.direct:
                solver = LinearSolverDirect(preconditioner, config=parameters)
            case _:
                raise ValueError(linear_solver_type)
        return solver

    def assemble_preconditioner(self, preconditioner_config: dict) -> Preconditioner:
        preconditioner_type, preconditioner_config = list(
            preconditioner_config.items()
        )[0]
        params = preconditioner_config.get(DecisionNodeNames.parameters_picker, {})
        match preconditioner_type:
            case PreconditionerNames.amg:
                preconditioner = PreconditionerPetscAMG(params)
            case PreconditionerNames.ilu:
                preconditioner = PreconditionerPetscILU(params)
            case PreconditionerNames.none:
                preconditioner = NonePreconditioner(params)
            case _:
                raise ValueError(preconditioner_type)
        return preconditioner


# def assemble_splitting(
#     self, splitting_config: dict, problem_description: ProblemDescription
# ):
#     splitting_type, splitting_options = list(splitting_config.items())[0]


#     if splitting_type == SplittingNames.schur:
#         return SplittingSchurDiag(
#             linear_solver_primary=linear_solver_primary,
#             linear_solver_secondary=linear_solver_secondary,
#             problem_description=problem_description,
#             config=splitting_options,
#         )
#     elif splitting_type == SplittingNames.fixed_stress:
#         return SplittingFixedStress(
#             linear_solver_primary=linear_solver_primary,
#             linear_solver_secondary=linear_solver_secondary,
#             problem_description=problem_description,
#             config=splitting_options,
#         )
#     elif splitting_type == SplittingNames.undrained_split:
#         return SplittingUndrained(
#             linear_solver_primary=linear_solver_primary,
#             linear_solver_secondary=linear_solver_secondary,
#             problem_description=problem_description,
#             config=splitting_options,
#         )
#     elif splitting_type == SplittingNames.schur_cd:
#         return SplittingSchurCD(
#             linear_solver_primary=linear_solver_primary,
#             linear_solver_secondary=linear_solver_secondary,
#             problem_description=problem_description,
#             config=splitting_options,
#         )
#     elif splitting_type == SplittingNames.cpr:
#         return SplittingCPR(
#             linear_solver_primary=linear_solver_primary,
#             linear_solver_secondary=linear_solver_secondary,
#             problem_description=problem_description,
#             config=splitting_options,
#         )

#     raise ValueError(splitting_type)
