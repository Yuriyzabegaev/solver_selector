from pytest import raises

from solver_selector.solver_space import (
    ConstantNode,
    KrylovSolverDecisionNode,
    KrylovSolverNode,
    NumericalParameter,
    ParametersNode,
    SolverConfigNode,
    SplittingNode,
)

EXPECTED_CONFIG = {
    "bicgstab": {
        "preconditioner": {
            "fixed_stress": {
                "primary": {
                    "none": {
                        "preconditioner": {"ilu": {"drop_tol": 1e-05, "param_1": 10}}
                    }
                },
                "secondary": {
                    "none": {
                        "preconditioner": {"ilu": {"drop_tol": 1e-05, "param_1": 10}}
                    }
                },
                "l_factor": 1,
            }
        }
    }
}


def test_solver_space():
    ilu_params = ParametersNode(
        {
            "drop_tol": NumericalParameter(
                bounds=(1e-8, 1e-2),
                default=1e-5,
                scale="log10",
            ),
            "param_1": 10,
        }
    )
    ilu = SolverConfigNode(name="ilu", children=[ilu_params])
    amg = ConstantNode(name="amg", params={"cycle": "w", "max_levels": 5})
    mono_precs = [ilu, amg]
    no_solver = KrylovSolverNode.from_preconditioners_list(mono_precs, name="none")
    fixed_stress_params = ParametersNode(
        {
            "l_factor": NumericalParameter(
                bounds=(0, 1),
                default=1,
            ),
        }
    )
    fixed_stress = SplittingNode.from_solvers_list(
        {"primary": [no_solver], "secondary": [no_solver]},
        name="fixed_stress",
        other_children=[fixed_stress_params],
    )

    precs = [ilu, amg, fixed_stress]
    gmres_params = ParametersNode(
        {
            "restart": NumericalParameter(
                bounds=(10, 100),
                default=30,
                dtype="int",
            ),
        }
    )
    gmres = KrylovSolverNode.from_preconditioners_list(
        precs, name="gmres", other_children=[gmres_params]
    )
    bicgstab = KrylovSolverNode.from_preconditioners_list(precs, name="bicgstab")
    direct = ConstantNode("direct")

    solver_space = KrylovSolverDecisionNode(options=[gmres, bicgstab, direct])

    all_solver_templates = solver_space.get_all_solvers()
    assert len(all_solver_templates) == 13

    solvers = []
    for solver_template in all_solver_templates:
        default_params = {}
        for node_id, param_space in solver_template.parameters.items():
            params = {}
            for param_name, param in param_space.numerical.items():
                params[param_name] = param.default
            default_params[node_id] = params

        solvers.append(solver_template.select_parameters(default_params))

    for solver in solvers:
        config = solver_space.config_from_decision(solver, optimized_only=True)
        config = solver_space.config_from_decision(solver, optimized_only=False)
        reproduced_solver = solver_space.decision_from_config(config)
        assert solver == reproduced_solver

    solver = solvers[8]
    config = solver_space.config_from_decision(solver)

    assert config == EXPECTED_CONFIG

    config_str = solver_space.format_config(config)
    expected = (
        "bicgstab - fixed_stress [primary - ilu [drop_tol=1e-05, param_1=10], secondary"
        " - ilu [drop_tol=1e-05, param_1=10], l_factor=1]"
    )
    assert config_str == expected

    nodes = solver_space.find_nodes_by_name("amg")
    assert len(nodes) == 6
    assert len(set(node._id for node in nodes)) == 6
    for node in nodes:
        assert node.name == "amg"

    some_node_id = nodes[-1]._id
    node = solver_space.find_node_by_id(some_node_id)
    assert node._id == some_node_id


def test_check_unique_ids():
    ilu = SolverConfigNode(name="ilu")
    with raises(ValueError):
        _ = SolverConfigNode(children=[ilu, ilu], name="head")


def test_splitting():
    subsolver1 = KrylovSolverNode([ConstantNode("amg")], name="gmres")
    subsolver2 = KrylovSolverNode([ConstantNode("ilu")], name="bicgstab")
    subsolver31 = ConstantNode("direct")
    subsolver32 = ConstantNode("lu")
    splitting = SplittingNode.from_solvers_list(
        {
            "subsolver1": [subsolver1],
            "subsolver2": [subsolver2],
            "subsolver3": [subsolver31, subsolver32],
        },
        name="fixed_stress",
    )
    all_solvers = splitting.get_all_solvers()
    assert len(all_solvers) == 2

    expected_configs = [
        {
            "fixed_stress": {
                "subsolver1": {"gmres": {"amg": {}}},
                "subsolver2": {"bicgstab": {"ilu": {}}},
                "subsolver3": {"direct": {}},
            }
        },
        {
            "fixed_stress": {
                "subsolver1": {"gmres": {"amg": {}}},
                "subsolver2": {"bicgstab": {"ilu": {}}},
                "subsolver3": {"lu": {}},
            }
        },
    ]

    for solver_template in all_solvers:
        solver = solver_template.use_defaults()
        config = splitting.config_from_decision(solver)
        expected_configs.remove(config)


if __name__ == "__main__":
    test_solver_space()
    test_check_unique_ids()
    test_splitting()
