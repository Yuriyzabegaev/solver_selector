from solver_selector.solver_space import (
    SplittingNode,
    KrylovSolverDecisionNode,
    SolverConfigNode,
    NumericalParameter,
    ParametersNode,
    ConstantNode,
)
from solvers_common import (
    DirectSolverNode,
    LinearSolver,
    LinearSolverNames,
    SplittingSchur,
    SolverAssembler,
    Preconditioner,
)

from typing import Sequence, Literal, TYPE_CHECKING


if TYPE_CHECKING:
    from mandel_model import MandelSimulationModel


#  ------------------- Solver space -------------------


class SplittingNames:
    fixed_stress = "splitting_fixed_stress"

    primary_subsolver = "primary"
    secondary_subsolver = "secondary"


class FixedStressNode(SolverConfigNode):
    type = SplittingNames.fixed_stress

    def __init__(
        self,
        solver_nodes: dict[str, Sequence[SolverConfigNode]],
        l_factor: float | NumericalParameter,
    ) -> None:
        self.l_factor = l_factor
        self.solver_nodes = solver_nodes
        splitting = SplittingNode.from_solvers_list(
            solver_nodes=solver_nodes,
            other_children=[
                ParametersNode({"l_factor": self.l_factor}),
                ConstantNode("primary_variable", "displacement"),
                ConstantNode("method", "upper"),
            ],
        )
        super().__init__(children=[splitting])

    def copy(self):
        return type(self)(
            solver_nodes=self.solver_nodes,
            l_factor=self.l_factor,
        )


def make_mandel_solver_space(l_factor: Literal["0", "1", "dynamic"]):
    if l_factor == "0":
        l_factor = 0
    elif l_factor == "1":
        l_factor = 1
    elif l_factor == "dynamic":
        l_factor = NumericalParameter(bounds=(0, 1), default=0, is_optimized=True)

    prec_primary = DirectSolverNode()
    prec_secondary = DirectSolverNode()
    # prec_secondary = LinearSolverNode2.from_preconditioners_list(
    #     name=LinearSolverNames.none,
    #     preconditioners=[NodePreconditionerAMG(config={'max_iter': 1})],
    # )
    # prec_secondary = LinearSolverNode2.from_preconditioners_list(
    #     name=LinearSolverNames.none,
    #     preconditioners=[NodePreconditionerAMG(config={'max_iter': 10})],
    # )

    return KrylovSolverDecisionNode.from_preconditioners_list(
        krylov_solver_name=LinearSolverNames.gmres,
        preconditioners=[
            FixedStressNode(
                {
                    SplittingNames.primary_subsolver: [prec_primary],
                    SplittingNames.secondary_subsolver: [prec_secondary],
                },
                l_factor=l_factor,
            ),
        ],
        other_children=[ParametersNode({"tol": 1e-6, "restart": 30, "maxiter": 10})],
    )


#  ------------------- Solvers implementations -------------------


class SplittingFixedStress(SplittingSchur):
    def __init__(
        self,
        mandel_problem: "MandelSimulationModel",
        linear_solver_primary: LinearSolver,
        linear_solver_secondary: LinearSolver,
        config: dict,
    ):
        self.simulation: "MandelSimulationModel" = mandel_problem
        for param in ["primary_variable", "l_factor"]:
            assert param in config
        super().__init__(linear_solver_primary, linear_solver_secondary, config)

    def update(self, mat):
        super().update(mat)
        prim_ind, sec_ind = self.simulation.get_primary_secondary_indices(
            primary_variable=self.config["primary_variable"]
        )
        self.primary_ind = prim_ind
        self.secondary_ind = sec_ind

        mat_00 = mat[prim_ind[:, None], prim_ind]
        mat_01 = mat[prim_ind[:, None], sec_ind]
        mat_10 = mat[sec_ind[:, None], prim_ind]
        mat_11 = mat[sec_ind[:, None], sec_ind]

        l_factor = self.config["l_factor"]
        stab = self.simulation.get_fixed_stress_stabilization(l_factor)
        mat_S = mat_11 + stab

        self.linear_solver_primary.update(mat_00)
        self.linear_solver_secondary.update(mat_S)
        self.mat_01 = mat_01
        self.mat_10 = mat_10


#  ------------------- Solver assembling routines -------------------


class MandelSolverAssembler(SolverAssembler):
    def assemble_preconditioner(
        self,
        preconditioner_config: dict,
    ) -> Preconditioner:
        preconditioner_type, preconditioner_params = list(
            preconditioner_config.items()
        )[0]
        if preconditioner_type == SplittingNames.fixed_stress:
            primary = SplittingNames.primary_subsolver
            secondary = SplittingNames.secondary_subsolver
            linear_solver_primary = self.assemble_linear_solver(
                linear_solver_config=preconditioner_params[primary]
            )
            linear_solver_secondary = self.assemble_linear_solver(
                linear_solver_config=preconditioner_params[secondary]
            )
            return SplittingFixedStress(
                mandel_problem=self.simulation,
                linear_solver_primary=linear_solver_primary,
                linear_solver_secondary=linear_solver_secondary,
                config=preconditioner_params,
            )
        return super().assemble_preconditioner(preconditioner_config)
