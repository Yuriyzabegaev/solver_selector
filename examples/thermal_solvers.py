from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
from solvers_common import (
    LinearSolver,
    LinearSolverNames,
    NodePreconditionerAMG,
    NodePreconditionerILU,
    Preconditioner,
    SolverAssembler,
    SplittingSchur,
)

from solver_selector.solver_space import (
    ConstantNode,
    KrylovSolverDecisionNode,
    KrylovSolverNode,
    ParametersNode,
    SolverConfigNode,
    SplittingNode,
)

if TYPE_CHECKING:
    from thermal_model import ThermalSimulationModel


#  ------------------- Solver space -------------------


class SplittingNames:
    cpr = "cpr"
    schur_cd = "schur_cd"

    primary_subsolver = "primary"
    secondary_subsolver = "secondary"


class SplittingNodeCPR(SolverConfigNode):
    type = SplittingNames.cpr

    def __init__(
        self,
        primary_variable: Optional[
            Literal["temperature", "pressure"] | SolverConfigNode
        ] = None,
    ):
        solver_nodes = {
            SplittingNames.primary_subsolver: [
                KrylovSolverNode.from_preconditioners_list(
                    [NodePreconditionerAMG()],
                    name=LinearSolverNames.none,
                )
            ],
            SplittingNames.secondary_subsolver: [
                KrylovSolverNode.from_preconditioners_list(
                    [NodePreconditionerILU()],
                    name=LinearSolverNames.none,
                )
            ],
        }
        if primary_variable is None:
            primary_variable = "pressure"
        if isinstance(primary_variable, str):
            primary_variable = ConstantNode("primary_variable", params=primary_variable)
        self.primary_variable: SolverConfigNode = primary_variable

        self.splitting = SplittingNode.from_solvers_list(solver_nodes=solver_nodes)
        super().__init__(children=[self.splitting, self.primary_variable])

    def copy(self):
        return SplittingNodeCPR(
            primary_variable=self.primary_variable.copy(),
        )


class SplittingNodeSchurCD(SolverConfigNode):
    type = SplittingNames.schur_cd

    def __init__(self) -> None:
        amg = KrylovSolverDecisionNode.from_preconditioners_list(
            krylov_solver_name=LinearSolverNames.none,
            preconditioners=[NodePreconditionerAMG()],
        )
        solver_nodes = {
            SplittingNames.primary_subsolver: [amg],
            SplittingNames.secondary_subsolver: [amg],
        }
        splitting = SplittingNode.from_solvers_list(solver_nodes=solver_nodes)

        super().__init__(
            children=[
                splitting,
                ConstantNode("primary_variable", "pressure"),
                ConstantNode("method", "full"),
            ],
        )

    def copy(self):
        return SplittingNodeSchurCD()


def make_thermal_solver_space(solver: Literal["schur", "cpr", "dynamic"]):
    schur = SplittingNodeSchurCD()
    cpr = SplittingNodeCPR(primary_variable="pressure")
    if solver == "schur":
        precs = [schur]
    elif solver == "cpr":
        precs = [cpr]
    elif solver == "dynamic":
        precs = [schur, cpr]
    else:
        raise ValueError(solver)
    return KrylovSolverDecisionNode.from_preconditioners_list(
        krylov_solver_name=LinearSolverNames.gmres,
        preconditioners=precs,
        other_children=[ParametersNode({"tol": 1e-10, "restart": 30, "maxiter": 10})],
    )


#  ------------------- Solvers implementations -------------------


class SplittingCPR(Preconditioner):
    def __init__(
        self,
        linear_solver_primary: LinearSolver,
        linear_solver_secondary: LinearSolver,
        thermal_problem: "ThermalSimulationModel",
        config: Optional[dict] = None,
    ):
        for param in ["primary_variable"]:
            assert param in config
        self.linear_solver_primary: LinearSolver = linear_solver_primary
        self.linear_solver_secondary: LinearSolver = linear_solver_secondary
        self.thermal_problem: "ThermalSimulationModel" = thermal_problem
        super().__init__(config=config)

    def update(self, mat):
        super().update(mat)
        prim_ind, sec_ind = self.thermal_problem.get_primary_secondary_indices(
            self.config["primary_variable"]
        )
        self.primary_ind = prim_ind
        mat_00 = mat[prim_ind[:, None], prim_ind]
        self.linear_solver_primary.update(mat_00)
        self.linear_solver_secondary.update(mat)

    def apply(self, b):
        rhs_0 = b[self.primary_ind]

        sol_0, _ = self.linear_solver_primary.solve(rhs_0)
        x_new = np.zeros_like(b)
        x_new[self.primary_ind] = sol_0
        tmp = b - self.mat.dot(x_new)
        sol_1, _ = self.linear_solver_secondary.solve(tmp)
        x_new += sol_1
        return x_new


class SplittingSchurCD(SplittingSchur):
    def __init__(
        self,
        thermal_problem: "ThermalSimulationModel",
        linear_solver_primary: LinearSolver,
        linear_solver_secondary: LinearSolver,
        config: dict,
    ):
        self.thermal_problem: "ThermalSimulationModel" = thermal_problem
        for param in ["primary_variable"]:
            assert param in config
        super().__init__(linear_solver_primary, linear_solver_secondary, config)

    def update(self, mat):
        super().update(mat)
        prim_ind, sec_ind = self.thermal_problem.get_primary_secondary_indices(
            self.config["primary_variable"]
        )
        self.primary_ind = prim_ind
        self.secondary_ind = sec_ind

        mat_00 = mat[prim_ind[:, None], prim_ind]
        mat_01 = mat[prim_ind[:, None], sec_ind]
        mat_10 = mat[sec_ind[:, None], prim_ind]
        mat_S = self.thermal_problem.get_schur_cd_approximation(sec_ind)

        self.linear_solver_primary.update(mat_00)
        self.linear_solver_secondary.update(mat_S)
        self.mat_01 = mat_01
        self.mat_10 = mat_10


#  ------------------- Solver assembling routines -------------------


class ThermalSolverAssembler(SolverAssembler):
    def assemble_preconditioner(
        self,
        preconditioner_config: dict,
    ) -> Preconditioner:
        preconditioner_type, preconditioner_params = list(
            preconditioner_config.items()
        )[0]

        if preconditioner_type not in [SplittingNames.cpr, SplittingNames.schur_cd]:
            return super().assemble_preconditioner(preconditioner_config)

        primary = SplittingNames.primary_subsolver
        secondary = SplittingNames.secondary_subsolver
        linear_solver_primary = self.assemble_linear_solver(
            linear_solver_config=preconditioner_params[primary]
        )
        linear_solver_secondary = self.assemble_linear_solver(
            linear_solver_config=preconditioner_params[secondary]
        )

        if preconditioner_type == SplittingNames.cpr:
            return SplittingCPR(
                thermal_problem=self.simulation,
                linear_solver_primary=linear_solver_primary,
                linear_solver_secondary=linear_solver_secondary,
                config=preconditioner_params,
            )
        elif preconditioner_type == SplittingNames.schur_cd:
            return SplittingSchurCD(
                thermal_problem=self.simulation,
                linear_solver_primary=linear_solver_primary,
                linear_solver_secondary=linear_solver_secondary,
                config=preconditioner_params,
            )
        else:
            raise ValueError(preconditioner_type)
