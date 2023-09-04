import numpy as np


from solver_selector.solver_space import (
    KrylovSolverNode,
    SolverConfigNode,
    ParametersNode,
    NumericalParameter,
)
from solver_selector.performance_predictor import (
    ParametersSpace,
    PerformancePredictorEpsGreedy,
    DEFAULT_EXPECTATION,
)

from tests_common import DummpyProblemContext, generate_synthetic_data


def make_solver_template():
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
    gmres_params = ParametersNode(
        {
            "restart": NumericalParameter(
                bounds=(10, 100),
                default=30,
                dtype="int",
            ),
        }
    )
    gmres = KrylovSolverNode(children=[ilu, gmres_params], name="gmres")
    solver_templates = gmres.get_all_solvers()
    assert len(solver_templates) == 1
    return solver_templates[0]


def test_parameters_space():
    params_space = ParametersSpace(make_solver_template())
    assert params_space.param_names == ["drop_tol", "param_1", "restart"]
    assert params_space.bounds == [(1e-08, 0.01), (10, 10), (10, 100)]
    assert params_space.defaults == [1e-05, 10, 30]
    assert params_space.node_ids[0] == params_space.node_ids[1]
    assert params_space.node_ids[0] != params_space.node_ids[2]
    assert params_space.is_optimized == [True, False, True]

    x_space = params_space.make_parameters_grid(num_samples=30)
    assert x_space.shape == (900, 2)

    # Now the selection algorithm chooses one parameters configuration.
    parameters_array = x_space[42]
    decision = params_space.decision_from_array(parameters_array)
    reconstructed_array = params_space.array_from_decision(decision)

    assert np.all(reconstructed_array == parameters_array)


def test_performance_predictor():
    solver_template = make_solver_template()
    performance_predictor = PerformancePredictorEpsGreedy(
        decision_template=solver_template,
        samples_before_fit=5,
        exploration=0,
        exploration_rate=0,
    )
    context = DummpyProblemContext()

    np.random.seed(42)  # It is not initialized, so selects randomly.
    prediction = performance_predictor.select_solver_parameters(context)
    assert prediction.score == DEFAULT_EXPECTATION

    # Generating synthetic data


    # Online learning process
    np.random.seed(42)  # Maybe something random happens inside sklearn.
    simulation_data = generate_synthetic_data(prediction=prediction, seed=0)
    for time_step_data in simulation_data:
        performance_predictor.online_update(time_step_data)

    assert len(performance_predictor.memory_contexts) == 10
    assert len(performance_predictor.memory_rewards) == 10
    assert performance_predictor.is_initialized

    prediction = performance_predictor.select_solver_parameters(context)
    # It is initialized, exploration is off, so it must select based on the prediction.
    assert prediction.score != DEFAULT_EXPECTATION

    # Now the same for offline training. Result should be the same for online / offline.
    performance_predictor_offline = PerformancePredictorEpsGreedy(
        decision_template=solver_template,
        samples_before_fit=5,
        exploration=0,
        exploration_rate=0,
    )
    np.random.seed(42)  # Maybe something random happens inside sklearn.
    performance_predictor_offline.offline_update(simulation_data)
    assert len(performance_predictor_offline.memory_contexts) == 10
    assert len(performance_predictor_offline.memory_rewards) == 10
    assert performance_predictor_offline.is_initialized

    prediction_offline = performance_predictor_offline.select_solver_parameters(context)
    assert prediction_offline == prediction, "The same for the same training data."


if __name__ == "__main__":
    test_parameters_space()
    test_performance_predictor()
