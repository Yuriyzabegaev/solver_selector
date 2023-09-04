from mandel_solvers import make_mandel_solver_space, MandelSolverAssembler
from thermal_solvers import make_thermal_solver_space, ThermalSolverAssembler


def test_mandel_solvers():
    make_mandel_solver_space(l_factor=0)
    make_mandel_solver_space(l_factor=1)
    solver_space = make_mandel_solver_space(l_factor="dynamic")
    solver_templates = solver_space.get_all_solvers()
    assert len(solver_templates) == 1
    decision = solver_templates[0].use_defaults()
    config = solver_space.config_from_decision(decision)
    solver = MandelSolverAssembler(simulation=None).assemble_linear_solver(config)


def test_thermal_solvers():
    make_thermal_solver_space(solver="schur")
    make_thermal_solver_space(solver="cpr")
    solver_space = make_thermal_solver_space(solver="dynamic")
    solver_templates = solver_space.get_all_solvers()
    assert len(solver_templates) == 2
    for solver_template in solver_templates:
        decision = solver_template.use_defaults()
        config = solver_space.config_from_decision(decision)
        solver = ThermalSolverAssembler(simulation=None).assemble_linear_solver(config)


if __name__ == "__main__":
    test_mandel_solvers()
    test_thermal_solvers()
