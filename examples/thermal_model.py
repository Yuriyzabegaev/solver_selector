from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import porepy as pp
from porepy.models.mass_and_energy_balance import MassAndEnergyBalance, energy, mass
from porepy.numerics.ad.operators import Scalar, TimeDependentDenseArray
from porepy_common import PorepyNewtonSolver, PorepySimulation
from thermal_solvers import ThermalSolverAssembler

from solver_selector.data_structures import ProblemContext, SolverSelectionData
from solver_selector.simulation_runner import Solver

PERMEABILITY = 1000

DAYS = 24 * 3600
YEARS = DAYS * 365

pumping_schedule = [
    [0, 100 * DAYS],
    [50 * YEARS, 50 * YEARS + 100 * DAYS],
    [100 * YEARS, 100 * YEARS + 100 * DAYS],
    # [150 * YEARS, 150 * YEARS + 100 * DAYS],
    # [200 * YEARS, 200 * YEARS + 100 * DAYS],
    # [250 * YEARS, 250 * YEARS + 100 * DAYS],
]
T_END = 300 * YEARS
dt = 1.4e1


def cond_(x, a, b):
    x += 1e-50
    cond_a = pp.ad.maximum(pp.ad.sign(x), 0)
    cond_b = pp.ad.maximum(-pp.ad.sign(x), 0)
    if not isinstance(a, pp.ad.AdArray):
        return b * cond_b + a * cond_a
    return a * cond_a + b * cond_b


cond = pp.ad.Function(cond_, name="condition", array_compatible=True)
ad_max = pp.ad.Function(pp.ad.maximum, name="max", array_compatible=True)
ad_min = pp.ad.Function(
    lambda x, y: -pp.ad.maximum(-x, -y), name="min", array_compatible=True
)


class ThermalBase(MassAndEnergyBalance):
    def __init__(self, params: dict | None = None) -> None:
        spe10_phi_path = Path(params["spe10_phi"])
        spe10_perm_path = Path(params["spe10_perm"])
        self._perm = np.load(spe10_perm_path).T * PERMEABILITY * 1e6  # From m^2 to mm^2
        self._phi = np.load(spe10_phi_path).T
        self._phi = np.maximum(self._phi, 1e-10)
        # self._phi = self._phi[:10, :10]
        # self._perm = self._perm[:10, :10]
        self._shape = self._phi.shape
        super().__init__(params)

    def pressure_exponential(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Exponential term in the fluid density as a function of pressure.

        Extracted as a separate method to allow for easier combination with temperature
        dependent fluid density.

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Exponential term in the fluid density as a function of pressure.

        """
        exp = pp.ad.Function(pp.ad.exp, "density_exponential")
        dp = self.pressure(subdomains) - 1.01325e5 * 1e-6
        c = self.fluid_compressibility(subdomains)
        return exp(c * dp)

    def temperature_exponential(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Exponential term in the fluid density as a function of pressure.

        Extracted as a separate method to allow for easier combination with temperature
        dependent fluid density.

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Exponential term in the fluid density as a function of pressure.

        """
        exp = pp.ad.Function(pp.ad.exp, "density_exponential")
        dtemp = self.temperature(subdomains) - 288.7056
        return exp(Scalar(-1) * Scalar(self.fluid.thermal_expansion()) * dtemp)

    def fluid_viscosity_formula(self, T):
        API = 10.0
        A1 = -0.8021
        A2 = 23.8765
        A3 = 0.31458
        A4 = -9.21592
        Tf = Scalar(1.8) * (T - 273.15) + 32.0  # temperature in Fahrenheit
        return (Tf ** (A3 * API + A4) * 10.0 ** (A1 * API + A2)) * 1e-3

    def fluid_viscosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid viscosity [Pa s].

        Parameters:
            subdomains: List of subdomain grids. Not used in this implementation, but
                included for compatibility with other implementations.

        Returns:
            Operator for fluid viscosity, represented as an Ad operator. The value is
            picked from the fluid constants.

        """

        T = self.temperature(subdomains)
        val = self.fluid_viscosity_formula(T)
        val.set_name("fluid_viscosity")
        return val

    def fluid_enthalpy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid enthalpy [J*kg^-1*m^-nd].

        The enthalpy is computed as a perturbation from a reference temperature as
        .. math::
            h = c_p (T - T_0)

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the fluid enthalpy.

        """
        c = self.fluid_specific_heat_capacity(subdomains)
        enthalpy = c * self.temperature(subdomains)
        enthalpy.set_name("fluid_enthalpy")
        return enthalpy

    def solid_enthalpy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Solid enthalpy [J*kg^-1*m^-nd].

        The enthalpy is computed as a perturbation from a reference temperature as
        .. math::
            h = c_p (T - T_0)

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the solid enthalpy.

        """
        c = self.solid_specific_heat_capacity(subdomains)
        enthalpy = c * self.temperature(subdomains)
        enthalpy.set_name("solid_enthalpy")
        return enthalpy

    def fluid_internal_energy(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Internal energy of the fluid.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the fluid energy.

        """
        energy = (
            self.fluid_density(subdomains) * self.fluid_enthalpy(subdomains)
        ) * self.porosity(subdomains)
        energy.set_name("fluid_internal_energy")
        return energy

    def set_equations(self):
        """Set the equations for the fluid mass and energy balance problem.

        Call both parent classes' set_equations methods.

        """
        # Mass balance
        mass.MassBalanceEquations.set_equations(self)
        # Energy balance
        energy.EnergyBalanceEquations.set_equations(self)

    def create_variables(self) -> None:
        """Set the variables for the fluid mass and energy balance problem.

        Call both parent classes' set_variables methods.

        """
        # Energy balance
        mass.VariablesSinglePhaseFlow.create_variables(self)
        energy.VariablesEnergyBalance.create_variables(self)

    def porosity(self, subdomains):
        if len(subdomains) > 0:
            return pp.ad.DenseArray(self._phi.ravel(order="f"))
        return pp.ad.DenseArray(np.zeros(0))

    def permeability(self, subdomains):
        if len(subdomains) > 0:
            return pp.ad.DenseArray(self._perm.ravel(order="f"))
        return pp.ad.DenseArray(np.zeros(0))

    def set_domain(self) -> None:
        # size = 365.76 / self.units.m
        cell_size_x = 6.096 / self.units.m
        cell_size_y = 3.048 / self.units.m
        cell_size_z = 3.048 / self.units.m
        if len(self._shape) == 3:
            z_shape = self._shape[2]
        else:
            z_shape = 1
        self._domain = pp.Domain(
            {
                "xmin": 0,
                "xmax": cell_size_x * self._shape[0],
                "ymin": 0,
                "ymax": cell_size_y * self._shape[1],
                "zmin": 0,
                "zmax": cell_size_z * z_shape,
            }
        )

    def meshing_arguments(self) -> dict:
        return {
            "cell_size_x": 6.096 / self.units.m,
            "cell_size_y": 3.048 / self.units.m,
            "cell_size_z": 3.048 / self.units.m,
        }

    def initial_condition(self) -> None:
        super().initial_condition()
        vals = np.zeros(self.equation_system.num_dofs())
        vals[
            self.equation_system.dofs_of([self.pressure_variable])
        ] = self.fluid.pressure()
        vals[
            self.equation_system.dofs_of([self.temperature_variable])
        ] = self.fluid.temperature()
        for time_step_index in self.time_step_indices:
            self.equation_system.set_variable_values(
                values=vals, time_step_index=time_step_index
            )

    def before_nonlinear_loop(self) -> None:
        super().before_nonlinear_loop()
        state = self.equation_system.get_variable_values(time_step_index=0)
        self.equation_system.set_variable_values(state, iterate_index=0)

    def after_nonlinear_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        subdomains = self.mdg.subdomains()
        visc = self.fluid_viscosity(subdomains).evaluate(self.equation_system).val
        dens = self.fluid_density(subdomains).evaluate(self.equation_system).val
        print(f"visc min: {visc.min()}, max: {visc.max()}")
        print(f"rho min: {dens.min()}, max: {dens.max()}")
        super().after_nonlinear_convergence(solution, errors, iteration_counter)


class ThermalSource(ThermalBase):
    PRESSURE_INLET = 6.895e7 * 1e-6
    PRESSURE_OUTLET = 2.7579e7 * 1e-6

    TEMPERATURE_INLET = 422.039
    TEMPERATURE_OUTLET = 422.039

    RATE = 1e-3

    KEY_FLUID_SOURCE = "key_fluid_source"

    def get_inlet_cells(self, sd: pp.Grid):
        source_location: int = self.params['source_location']
        if source_location == 0:
            # high-perm region
            x_inj = 265
            y_inj = 260
        elif source_location == 1:
            x_inj = 140.0
            y_inj = 210.0
        else:
            raise ValueError
        x, y, _ = sd.cell_centers
        loc = np.argmin(np.sqrt((x - x_inj) ** 2 + (y - y_inj) ** 2))
        return loc

    def get_outlet_cells(self, sd: pp.Grid):
        source_location: int = self.params['source_location']
        if source_location == 0:
            x_inj = 140.0
            y_inj = 210.0
        elif source_location == 1:
            x_inj = 140.
            y_inj = 100.
        else:
            raise ValueError
        x, y, _ = sd.cell_centers
        loc = np.argmin(np.sqrt((x - x_inj) ** 2 + (y - y_inj) ** 2))
        return loc

    def is_pumping(self):
        is_pumping = False
        for start, end in pumping_schedule:
            if start <= self.time_manager.time < end:
                is_pumping = True
                print("SOURCE ON")
                break
        self.well_on = is_pumping
        return is_pumping

    def fluid_density_inlet(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        rho_ref = self.fluid.density()

        exp = pp.ad.Function(pp.ad.exp, "density_exponential")
        dtemp = self.TEMPERATURE_INLET - 288.7056
        temperature_exponential = exp(
            pp.ad.Scalar(-1 * self.fluid.thermal_expansion()) * dtemp
        )

        rho = (
            Scalar(rho_ref)
            * self.pressure_exponential(subdomains)
            * temperature_exponential
        )
        rho.set_name("fluid_density_inlet")
        return rho

    def peaceman_flow_rate(self, subdomains, p_max, rate_max):
        h = 5
        rw = 0.1
        Dx = 5
        Dy = 5
        Ky_div_Kx = 1
        Kx_div_Ky = 1
        Kx = self.permeability(subdomains)
        Ky = Kx
        mu = self.fluid_viscosity(subdomains)

        p = self.pressure(subdomains)

        ro = (
            0.28
            * np.sqrt(np.sqrt(Ky_div_Kx) * Dx**2 + np.sqrt(Kx_div_Ky) * Dy**2)
            / (Ky_div_Kx**0.25 + Kx_div_Ky**0.25)
        )
        Ke = (Kx * Ky) ** 0.5
        is_prod = rate_max < 0
        rate_max = pp.ad.Scalar(rate_max)
        p_max = pp.ad.Scalar(p_max)

        factor = Scalar(2) * np.pi * h * Ke / np.log(ro / rw) / mu

        if is_prod:
            dd = cond(p_max - p, pp.ad.Scalar(0), (p_max - p))
        else:
            dd = cond(p - p_max, pp.ad.Scalar(0), (p_max - p))

        rate = factor * dd

        if is_prod:
            rate = ad_max(rate, rate_max)
        else:
            rate = ad_min(rate, rate_max)
        return rate

    def fluid_rate_inlet(self, subdomains):
        return self.peaceman_flow_rate(
            subdomains, rate_max=self.RATE, p_max=self.PRESSURE_INLET
        )

    def fluid_rate_outlet(self, subdomains):
        return self.peaceman_flow_rate(
            subdomains, rate_max=-self.RATE, p_max=self.PRESSURE_OUTLET
        )

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        super_source = super().fluid_source(subdomains)
        if len(subdomains) == 0:
            return super_source

        outlets = []
        for sd in subdomains:
            vals = np.zeros(sd.num_cells)
            outlet_cells = self.get_outlet_cells(sd)
            vals[outlet_cells] = 1
            # vals *= sd.cell_volumes
            outlets.append(vals)
        outlets = (
            self.fluid_density(subdomains)
            * self.fluid_rate_outlet(subdomains)
            * np.concatenate(outlets)
        )

        inlets = []
        for sd in subdomains:
            vals = np.zeros(sd.num_cells)
            inlet_cells = self.get_inlet_cells(sd)
            vals[inlet_cells] = 1
            # vals *= sd.cell_volumes
            inlets.append(vals)

        inlets = (
            self.fluid_density_inlet(subdomains)
            * self.fluid_rate_inlet(subdomains)
            * np.concatenate(inlets)
        )

        source_switch = TimeDependentDenseArray(
            self.KEY_FLUID_SOURCE, domains=subdomains
        )

        return super_source + (outlets + inlets) * source_switch

    def energy_outlet(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        outlets = []
        for sd in subdomains:
            vals = np.zeros(sd.num_cells)
            outlet_cells = self.get_outlet_cells(sd)
            vals[outlet_cells] = 1
            # vals *= sd.cell_volumes
            outlets.append(vals)

        outlets = (
            self.fluid_density(subdomains)
            * np.concatenate(outlets)
            * self.fluid.specific_heat_capacity()
            * self.fluid_rate_outlet(subdomains)
            * self.temperature(subdomains)
        )
        return outlets

    def energy_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        super_source = super().energy_source(subdomains)
        if len(subdomains) == 0:
            return super_source

        # INLETS
        inlets = []
        for sd in subdomains:
            vals = np.zeros(sd.num_cells)
            inlet_cells = self.get_inlet_cells(sd)
            vals[inlet_cells] = self.TEMPERATURE_INLET
            # vals *= sd.cell_volumes
            inlets.append(vals)

        inlets = (
            self.fluid_density_inlet(subdomains)
            * np.concatenate(inlets)
            * self.fluid.specific_heat_capacity()
            * self.fluid_rate_inlet(subdomains)
        )

        # OUTLETS
        outlets = self.energy_outlet(subdomains)

        source_switch = TimeDependentDenseArray(
            self.KEY_FLUID_SOURCE, domains=subdomains
        )
        # source_switch = 1
        return super_source + (inlets + outlets) * source_switch

    def update_source_pumping(self, subdomains: list[pp.Grid]):
        res = np.zeros(subdomains[0].num_cells)
        res[:] = self.is_pumping()
        return res

    def update_time_dependent_ad_arrays(self) -> None:
        super().update_time_dependent_ad_arrays()
        for sd, data in self.mdg.subdomains(return_data=True):
            if self.KEY_FLUID_SOURCE in data[pp.ITERATE_SOLUTIONS]:
                # Copy old values from iterate to the solution.
                vals = pp.get_solution_values(
                    name=self.KEY_FLUID_SOURCE, data=data, iterate_index=0
                )
            else:
                # First time step.
                vals = self.update_source_pumping([sd])
            pp.set_solution_values(
                name=self.KEY_FLUID_SOURCE, values=vals, data=data, time_step_index=0
            )

            vals = self.update_source_pumping([sd])
            pp.set_solution_values(
                name=self.KEY_FLUID_SOURCE, values=vals, data=data, iterate_index=0
            )


class ThermalBC(ThermalBase):
    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        bounds = self.domain_boundary_sides(sd).all_bf
        return pp.BoundaryCondition(sd, bounds, "neu")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        bounds = self.domain_boundary_sides(sd).all_bf
        return pp.BoundaryCondition(sd, bounds, "neu")


class SchurCDApproximation(ThermalBase):
    def prepare_simulation(self) -> None:
        super().prepare_simulation()
        subdomains = self.mdg.subdomains()
        self.schur_ad = self.set_schur_approximation(subdomains)

    def _fluid_density_formula(self, subdomains, pressure, temperature):
        exp = pp.ad.Function(pp.ad.exp, "density_exponential_pressure")
        dp = pressure - 1.01325e5 * 1e-6
        c = self.fluid_compressibility(subdomains)
        pressure_exponential = exp(c * dp)

        exp = pp.ad.Function(pp.ad.exp, "density_exponential_temperature")
        dtemp = temperature - 288.7056
        temperature_exponential = exp(
            pp.ad.Scalar(-1 * self.fluid.thermal_expansion()) * dtemp
        )

        rho_ref = pp.ad.Scalar(self.fluid.density(), "reference_fluid_density")
        rho = rho_ref * pressure_exponential * temperature_exponential
        rho.set_name("fluid_density_from_pressure_and_temperature")
        return rho

    def set_schur_approximation(self, subdomains: pp.Grid) -> pp.ad.Operator:
        # - \nabla * (k_T \nabla)
        diffusion_flux = self.fourier_flux(subdomains)
        diffusion_flux.set_name("schur_diffusion_flux")

        # c_v \nabla * (\rho u)
        discr = self.enthalpy_discretization(subdomains)
        T_prev = self.temperature(subdomains).previous_iteration()
        p_prev = self.pressure(subdomains).previous_iteration()
        viscosity = self.fluid_viscosity_formula(T_prev)
        mobility = pp.ad.Scalar(1) / viscosity
        fluid_density = self._fluid_density_formula(subdomains, p_prev, T_prev)

        def enthalpy_dirichlet(boundary_grids):
            result = self.fluid_enthalpy(boundary_grids)
            result *= self.mobility_rho(boundary_grids)
            return result

        boundary_operator_enthalpy = (
            self._combine_boundary_operators(  # type: ignore[call-arg]
                subdomains=subdomains,
                dirichlet_operator=enthalpy_dirichlet,
                neumann_operator=self.enthalpy_flux,
                bc_type=self.bc_type_enthalpy_flux,
                name="bc_values_enthalpy",
            )
        )

        convection_flux = self.advective_flux(
            subdomains,
            self.fluid_enthalpy(subdomains) * mobility * fluid_density,
            discr,
            boundary_operator_enthalpy,
            self.interface_enthalpy_flux,
        )
        convection_flux.set_name("schur_convection_flux")

        # c_v * f_prod
        source = self.energy_outlet(subdomains)
        # NO HEATERS HERE!!!

        fluid_internal_energy = (
            fluid_density * self.fluid_enthalpy(subdomains)
        ) * self.porosity(subdomains)
        fluid_internal_energy.set_name("fluid_internal_energy")
        total_internal_energy = fluid_internal_energy + self.solid_internal_energy(
            subdomains
        )
        total_internal_energy.set_name("total_energy")
        accumulation = self.volume_integral(total_internal_energy, subdomains, dim=1)

        dt = self.ad_time_step
        div = pp.ad.Divergence(subdomains)
        result = accumulation / dt + div @ (convection_flux + diffusion_flux) - source

        result.set_name("schur_approximation_cd")
        return result


class ThermalSetup(SchurCDApproximation, ThermalBC, ThermalSource, ThermalBase):
    pass


fluid_constants = pp.FluidConstants(
    {
        "compressibility": 5.5e-10 * 1e6,
        "density": 999.0,
        "specific_heat_capacity": 2093.4,  # Вместимость
        "thermal_conductivity": 0.15,  # Diffusion coefficient
        "viscosity": 1,
        "pressure": 4.1369e7 * 1e-6,
        "temperature": 288.706,
        "thermal_expansion": 2.5e-4,  # Density(T)
    }
)
solid_constants = pp.SolidConstants(
    {
        "density": 2650.0,
        "specific_heat_capacity": 920.0,
        "thermal_conductivity": 1.7295772056,  # Diffusion coefficient
        "temperature": 288.706,
    }
)


@dataclass(frozen=True, kw_only=True, slots=True)
class ThermalContext(ProblemContext):
    mat_size: int
    time_step_value: float
    peclet_max: float
    peclet_mean: float
    cfl_max: float
    cfl_mean: float
    well_on: int
    inlet_rate: float
    outlet_rate: float

    def get_array(self):
        return np.array(
            [
                np.log(self.time_step_value),
                np.log(np.max([self.peclet_max, 1e-8])),
                np.log(np.max([self.peclet_mean, 1e-8])),
                np.log(np.max([self.inlet_rate, 1e-8])),
                np.log(np.max([self.outlet_rate, 1e-8])),
                np.log(self.mat_size),
                self.well_on,
            ],
            dtype=float,
        )


class ThermalSimulationModel(PorepySimulation):
    def __init__(self, porepy_setup: ThermalSetup):
        self.porepy_setup: ThermalSetup = porepy_setup
        self.solver_assembler = ThermalSolverAssembler(self)

    def compute_peclet(self):
        model = self.porepy_setup
        fourier_flux = (
            model.fourier_flux(model.mdg.subdomains())
            .evaluate(model.equation_system)
            .val
        )
        enthalpy_flux = (
            model.enthalpy_flux(model.mdg.subdomains())
            .evaluate(model.equation_system)
            .val
        )
        fourier_flux = np.maximum(fourier_flux, 1e-10)
        peclet_max = abs(enthalpy_flux / fourier_flux).max()
        peclet_mean = abs(enthalpy_flux / fourier_flux).mean()
        return peclet_max, peclet_mean

    def compute_cfl(self):
        setup = self.porepy_setup
        velosity = (
            (
                setup.darcy_flux(setup.mdg.subdomains())
                / setup.fluid_viscosity_formula(422)
            )
            .evaluate(setup.equation_system)
            .val
        )
        length = setup.mdg.subdomains()[0].face_areas
        time_step = setup.time_manager.dt
        CFL = velosity * time_step / length
        CFL_max = abs(CFL).max()
        CFL_mean = abs(CFL).mean()
        return CFL_max, CFL_mean

    def compute_source_rate(self):
        setup = self.porepy_setup
        subdomains = setup.mdg.subdomains()
        inlet_cells = setup.get_inlet_cells(subdomains[0])
        outlet_cells = setup.get_outlet_cells(subdomains[0])
        inlet_rate = (
            setup.fluid_rate_inlet(subdomains)
            .evaluate(setup.equation_system)
            .val[inlet_cells]
            .max()
        )
        outlet_rate = (
            setup.fluid_rate_outlet(subdomains)
            .evaluate(setup.equation_system)
            .val[outlet_cells]
            .min()
        )
        sin_factor = setup.well_on
        inlet_rate *= sin_factor
        outlet_cells *= sin_factor
        return abs(inlet_rate), abs(outlet_rate)

    def get_context(self):
        peclet_max, peclet_mean = self.compute_peclet()
        CFL_max, CFL_mean = self.compute_cfl()
        fluid_rate_inlet, fluid_rate_outlet = self.compute_source_rate()
        return ThermalContext(
            mat_size=self.porepy_setup.equation_system._variable_num_dofs.sum(),
            time_step_value=self.porepy_setup.time_manager.dt,
            peclet_max=peclet_max,
            peclet_mean=peclet_mean,
            cfl_max=CFL_max,
            cfl_mean=CFL_mean,
            inlet_rate=fluid_rate_inlet,
            outlet_rate=fluid_rate_outlet,
            well_on=self.porepy_setup.well_on,
        )

    def assemble_solver(self, solver_config: dict) -> Solver:
        linear_solver = self.solver_assembler.assemble_linear_solver(solver_config)
        return PorepyNewtonSolver(linear_solver)

    def get_primary_secondary_indices(
        self, primary_variable: str
    ) -> tuple[np.ndarray, np.ndarray]:
        temperature = np.concatenate(
            [
                self.porepy_setup.equation_system.assembled_equation_indices[i]
                for i in (
                    "energy_balance_equation",
                    "interface_fourier_flux_equation",
                    "interface_enthalpy_flux_equation",
                    "well_enthalpy_flux_equation",
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

        if primary_variable == "temperature":
            return temperature, pressure
        elif primary_variable == "pressure":
            return pressure, temperature
        raise ValueError(f"{primary_variable=}")

    def get_schur_cd_approximation(self, indices: np.ndarray):
        schur_ad = self.porepy_setup.schur_ad.evaluate(
            self.porepy_setup.equation_system
        )
        return schur_ad.jac[:, indices]

    def after_time_step_success(self, solver_selection_data: SolverSelectionData):
        # Custom time step increasing scheme to follow Roy (2019).
        super().after_time_step_success(solver_selection_data)

        model = self.porepy_setup

        current_nits = len(solver_selection_data.nonlinear_solver_stats.iterations)

        if current_nits < 6:
            factor = 1 + min(1.0, (6 - current_nits) ** 2 / 3**2)
            model.time_manager.dt *= factor
        elif current_nits > 9:
            factor = 1 - min(1.0, (current_nits - 9) ** 2 / 4**2) / 2
            model.time_manager.dt *= factor

        model.time_manager._correction_based_on_dt_max()
        model.time_manager._correction_based_on_dt_min()
        model.time_manager._correction_based_on_schedule()

    def after_time_step_failure(self, solver_selection_data: SolverSelectionData):
        print("Time step failed")
        self.porepy_setup.time_manager.dt /= 2


def make_thermal_setup(
    model_size: Literal["small", "medium", "large"],
    source_location: Literal[0, 1] = 0,
) -> ThermalSimulationModel:
    base_path = Path(__file__).parent / "spe10_data"
    if model_size == "small":
        spe10_phi = base_path / "spe10_l0_120_phi.npy"
        spe10_perm = base_path / "spe10_l0_120_perm.npy"
    elif model_size == "medium":
        spe10_phi = base_path / "spe10_l0_220_phi.npy"
        spe10_perm = base_path / "spe10_l0_220_perm.npy"
    elif model_size == "large":
        spe10_phi = base_path / "spe10_l0_mirrored_phi.npy"
        spe10_perm = base_path / "spe10_l0_mirrored_perm.npy"
    else:
        raise ValueError(model_size)

    schedule = np.array(np.array(pumping_schedule).flatten().tolist() + [T_END])
    # schedule = [0, 10 * dt]
    # schedule = [0, dt]
    porepy_setup = ThermalSetup(
        {
            "material_constants": {
                "fluid": fluid_constants,
                "solid": solid_constants,
            },
            "spe10_phi": spe10_phi,
            "spe10_perm": spe10_perm,
            "time_manager": pp.TimeManager(
                schedule=schedule,
                dt_init=dt,
                constant_dt=False,
                dt_min_max=(dt, T_END / 10),
            ),
            "source_location": source_location,
        }
    )
    porepy_setup.prepare_simulation()
    return ThermalSimulationModel(porepy_setup=porepy_setup)
