import numpy as np
from tests_common import DummpyProblemContext, generate_synthetic_data

from solver_selector.data_structures import SolverSelectionData
from solver_selector.performance_predictor import (
    PerformancePredictor,
    PerformancePredictorEpsGreedy,
)
from solver_selector.solver_selector import SolverSelector
from solver_selector.solver_space import (
    ForkNode,
    KrylovSolverNode,
    NumericalParameter,
    ParametersNode,
    SolverConfigNode,
)


def make_solver_space():
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
    gmres = KrylovSolverNode.from_preconditioners_list(
        preconditioners=[ilu], other_children=[gmres_params], name="gmres"
    )
    bicgstab = KrylovSolverNode(children=[ilu], name="bicgstab")
    solver_space = ForkNode([gmres, bicgstab])
    assert len(solver_space.get_all_solvers()) == 2
    return solver_space


def make_solver_selector(solver_space: SolverConfigNode):
    all_solvers = solver_space.get_all_solvers()
    predictors: list[PerformancePredictor] = []
    for solver_template in all_solvers:
        predictors.append(
            PerformancePredictorEpsGreedy(
                decision_template=solver_template, exploration=0
            )
        )
    return SolverSelector(
        solver_space=solver_space,
        predictors=predictors,
    )


def test_sovler_selector():
    solver_space = make_solver_space()
    solver_selector = make_solver_selector(solver_space)
    context = DummpyProblemContext()

    # Synthetic data for each solver
    dataset: list[SolverSelectionData] = []
    for i, predictor in enumerate(solver_selector.predictors):
        prediction = predictor.select_solver_parameters(context)
        dataset.extend(
            generate_synthetic_data(
                prediction=prediction,
                config=solver_space.config_from_decision(prediction.decision),
                seed=i * 10,
            )
        )

    # Online learning
    np.random.seed(42)  # Randomness in sklearn
    for time_step in dataset:
        solver_selector.learn_performance_online(selection_data=time_step)
    prediction = solver_selector.select_solver(context)

    # Offline learning
    new_solver_space = make_solver_space()
    solver_selector_offline = make_solver_selector(new_solver_space)
    np.random.seed(42)  # Randomness in sklearn
    solver_selector_offline.learn_performance_offline(dataset)
    prediction_offline = solver_selector_offline.select_solver(context)

    assert prediction.score == prediction_offline.score
    config = solver_space.config_from_decision(
        prediction.decision
    )
    new_config = new_solver_space.config_from_decision(prediction_offline.decision)
    assert config == new_config


if __name__ == "__main__":
    test_sovler_selector()
