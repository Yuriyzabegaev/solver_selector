from solver_selector.solver_space import (
    KrylovSolverNode,
    KrylovSolverDecisionNode,
    SolverConfigNode,
    ConstantNode,
    SplittingNode,
    ParametersNode,
    NumericalParameter,
)


EXPECTED_CONFIG = {
    "linear solver": {
        "bicgstab": {
            "preconditioner": {
                "fixed_stress": {
                    "primary": {
                        "no_solver": {
                            "preconditioner": {
                                "ilu": {
                                    "parameters_node": {
                                        "drop_tol": 1e-05,
                                        "param_1": 10,
                                    }
                                }
                            }
                        }
                    },
                    "secondary": {
                        "no_solver": {
                            "preconditioner": {
                                "ilu": {
                                    "parameters_node": {
                                        "drop_tol": 1e-05,
                                        "param_1": 10,
                                    }
                                }
                            }
                        }
                    },
                    "parameters_node": {"l_factor": 1},
                }
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
    no_solver = KrylovSolverNode.from_preconditioners_list(mono_precs, name="no_solver")
    fixed_stress_params = ParametersNode(
        {
            "l_factor": NumericalParameter(
                bounds=(0, 1),
                default=1,
            ),
        }
    )
    fixed_stress = SplittingNode.from_solvers_list(
        [no_solver], name="fixed_stress", other_children=[fixed_stress_params]
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
    expected = "bicgstab - fixed_stress [primary - no_solver - ilu [drop_tol=1e-05, param_1=10], secondary - no_solver - ilu [drop_tol=1e-05, param_1=10], l_factor=1]"
    assert config_str == expected

    nodes = solver_space.find_nodes_by_name("amg")
    assert len(nodes) == 6
    assert len(set(node._id for node in nodes)) == 6
    for node in nodes:
        assert node.name == "amg"

    some_node_id = nodes[-1]._id
    node = solver_space.find_node_by_id(some_node_id)
    assert node._id == some_node_id


if __name__ == "__main__":
    test_solver_space()
