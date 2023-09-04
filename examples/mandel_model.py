import porepy as pp
import numpy as np
import scipy.sparse
from mandel_biot import MandelSetup as MandelSetup_
from scipy.sparse import csr_matrix
from solver_selector.simulation_runner import Solver
from solver_selector.solver_space import DecisionNodeNames
from solver_selector.data_structures import ProblemContext, SolverSelectionData
from mandel_solvers import MandelSolverAssembler
from porepy_common import PorepyNewtonSolver, PorepySimulation
from dataclasses import dataclass


class MandelSetup(MandelSetup_):
    def save_data_time_step(self) -> None:
        # Prevent writing data that we don't need.
        return pp.DataSavingMixin.save_data_time_step(self)


@dataclass(frozen=True, kw_only=True, slots=True)
class MandelContext(ProblemContext):
    mat_size: int
    time_step_value: float
    cfl_max: float
    cfl_mean: float

    def get_array(self):
        return np.array(
            [
                np.log(self.time_step_value),
                np.log(self.mat_size),
                np.log(self.cfl_max),
                np.log(self.cfl_mean),
            ],
            dtype=float,
        )


class MandelSimulationModel(PorepySimulation):
    def __init__(self, porepy_setup: MandelSetup):
        self.porepy_setup: MandelSetup = porepy_setup
        self.solver_assembler = MandelSolverAssembler(self)

    def get_context(self) -> ProblemContext:
        CFL_max, CFL_mean = self._compute_cfl()
        return MandelContext(
            mat_size=self.porepy_setup.equation_system._variable_num_dofs.sum(),
            time_step_value=self.porepy_setup.time_manager.dt,
            cfl_max=CFL_max,
            cfl_mean=CFL_mean,
        )

    def assemble_solver(self, solver_config: dict) -> Solver:
        linear_solver = self.solver_assembler.assemble_linear_solver(solver_config)
        return PorepyNewtonSolver(linear_solver)

    def get_primary_secondary_indices(
        self, primary_variable: str
    ) -> tuple[np.ndarray, np.ndarray]:
        displacement = np.concatenate(
            [
                self.porepy_setup.equation_system.assembled_equation_indices[i]
                for i in (
                    "momentum_balance_equation",
                    "normal_fracture_deformation_equation",
                    "tangential_fracture_deformation_equation",
                    "interface_force_balance_equation",
                )
            ]
        )
        pressure = np.concatenate(
            [
                self.porepy_setup.equation_system.assembled_equation_indices[i]
                for i in (
                    "mass_balance_equation",
                    "interface_darcy_flux_equation",
                    "well_flux_equation",
                )
            ]
        )
        if primary_variable == "displacement":
            return displacement, pressure
        elif primary_variable == "pressure":
            return pressure, displacement
        raise ValueError(f"{primary_variable=}")

    def get_fixed_stress_stabilization(self, l_factor: float) -> csr_matrix:
        porepy_setup = self.porepy_setup
        mu_lame = porepy_setup.solid.shear_modulus()
        lambda_lame = porepy_setup.solid.lame_lambda()
        alpha_biot = porepy_setup.solid.biot_coefficient()
        dim = 2

        l_phys = alpha_biot**2 / (2 * mu_lame / dim + lambda_lame)
        l_min = alpha_biot**2 / (4 * mu_lame + 2 * lambda_lame)

        val = l_min * (l_phys / l_min) ** l_factor

        diagonal_approx = val
        subdomains = porepy_setup.mdg.subdomains()
        cell_volumes = subdomains[0].cell_volumes
        diagonal_approx *= cell_volumes

        density = (
            porepy_setup.fluid_density(subdomains)
            .evaluate(porepy_setup.equation_system)
            .val
        )
        diagonal_approx *= density

        dt = porepy_setup.time_manager.dt
        diagonal_approx /= dt

        return scipy.sparse.diags(diagonal_approx)


time_manager = pp.TimeManager(
    schedule=[0, 1e3],
    dt_init=10,
    constant_dt=True,
)

units = pp.Units(
    # m=1e-3,
    # kg=1e3,
)

ls = 1 / units.m  # length scaling
mesh_arguments = {
    "cell_size": 1 * ls,
}

mandel_solid_constants = {
    "lame_lambda": 1.65e9 * 1e-6,  # [MPa]
    "shear_modulus": 2.475e9 * 1e-6,  # [MPa]
    "specific_storage": 6.0606e-11 * 1e6,  # [MPa^-1]
    "permeability": 9.869e-14 * 1e6,  # [mm^2]
    "biot_coefficient": 1.0,  # [-]
}

mandel_fluid_constants = {
    "density": 1e3,  # [kg * m^-3]
    "viscosity": 1e-3,  # [Pa * s]
}


def make_mandel_setup() -> MandelSimulationModel:
    porepy_setup = MandelSetup(
        {
            "meshing_arguments": mesh_arguments,
            "time_manager": time_manager,
            "material_constants": {
                "solid": pp.SolidConstants(mandel_solid_constants),
                "fluid": pp.FluidConstants(mandel_fluid_constants),
            },
        }
    )

    porepy_setup.prepare_simulation()
    return MandelSimulationModel(porepy_setup)
