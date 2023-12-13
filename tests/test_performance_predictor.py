import warnings

import numpy as np
import pytest
from tests_common import DummpyProblemContext, generate_synthetic_data

from solver_selector.performance_predictor import (
    DEFAULT_EXPECTATION,
    ParametersSpace,
    make_performance_predictor,
)
from solver_selector.solver_space import (
    KrylovSolverNode,
    NumericalParameter,
    ParametersNode,
    SolverConfigNode,
)


def make_solver_space() -> SolverConfigNode:
    ilu_params = ParametersNode(
        {
            "drop_tol": NumericalParameter(
                bounds=(1e-8, 1e-2),
                default=1e-5,
                # scale="log10",
            ),
            "param_1": 10,
        }
    )
    ilu = SolverConfigNode(name="ilu", children=[ilu_params])
    gmres_params = ParametersNode(
        {
            "restart": NumericalParameter(
                bounds=(10, 100),
                default=30,
                dtype="int",
            ),
        }
    )
    return KrylovSolverNode(children=[ilu, gmres_params], name="gmres")


def test_parameters_space():
    solver_space = make_solver_space()
    solver_templates = solver_space.get_all_solvers()
    assert len(solver_templates) == 1
    params_space = ParametersSpace(solver_templates[0])
    assert params_space.names == ["drop_tol", "param_1", "restart"]
    assert len(params_space.numerical_params) == len(params_space.node_ids) == 3
    assert params_space.node_ids[0] == params_space.node_ids[1]
    assert params_space.node_ids[0] != params_space.node_ids[2]

    x_space = params_space.make_parameters_grid(num_samples=30)
    assert x_space.shape == (900, 2)

    # Now the selection algorithm chooses one parameters configuration.
    parameters_array = x_space[42]
    decision = params_space.decision_from_array(parameters_array)
    reconstructed_array = params_space.array_from_decision(decision)

    assert np.all(reconstructed_array == parameters_array)


@pytest.mark.parametrize(
    "params",
    [
        {
            "predictor": "eps_greedy",
            "regressor": "gradient_boosting",
            'exploration': 0,
        },
        {
            "predictor": "eps_greedy",
            "regressor": "mlp",
            'exploration': 0,
            'samples_before_fit': 10,
        },
        {
            "predictor": "gaussian_process",
        },
    ],
)
def test_performance_predictor(params):
    solver_space = make_solver_space()
    solver_templates = solver_space.get_all_solvers()
    assert len(solver_templates) == 1
    solver_template = solver_templates[0]
    performance_predictor = make_performance_predictor(
        params=params, solver_template=solver_template
    )
    context = DummpyProblemContext()

    np.random.seed(42)  # It is not initialized, so selects randomly.
    prediction = performance_predictor.select_solver_parameters(context)
    assert prediction.score == DEFAULT_EXPECTATION

    # Online learning process
    np.random.seed(42)  # Maybe something random happens inside sklearn.
    simulation_data = generate_synthetic_data(
        prediction=prediction,
        config=solver_space.config_from_decision(prediction.decision),
        seed=0,
    )
    for time_step_data in simulation_data:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            performance_predictor.online_update(time_step_data)

    assert len(performance_predictor.memory_contexts) == 10
    assert len(performance_predictor.memory_rewards) == 10
    assert performance_predictor.is_initialized

    prediction = performance_predictor.select_solver_parameters(context)
    # It is initialized, exploration is off, so it must select based on the prediction.
    assert prediction.score != DEFAULT_EXPECTATION

    # Now the same for offline training. Result should be the same for online / offline.
    performance_predictor_offline = make_performance_predictor(
        solver_template=solver_template, params=params
    )
    np.random.seed(42)  # Maybe something random happens inside sklearn.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        performance_predictor_offline.offline_update(simulation_data)
    assert len(performance_predictor_offline.memory_contexts) == 10
    assert len(performance_predictor_offline.memory_rewards) == 10
    assert performance_predictor_offline.is_initialized

    prediction_offline = performance_predictor_offline.select_solver_parameters(context)
    assert prediction_offline == prediction, "The same for the same training data."


if __name__ == "__main__":
    test_parameters_space()
    test_performance_predictor()
