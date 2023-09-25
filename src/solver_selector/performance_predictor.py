from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

from solver_selector.data_structures import (
    PerformancePredictionData,
    ProblemContext,
    SolverSelectionData,
    load_data,
)
from solver_selector.solver_space import Decision, DecisionTemplate, number

DEFAULT_EXPECTATION = 100.0


class ParametersSpace:
    """An auxilary structure to represent the space of parameters within one algorithm.
    All the combinations of the parameters are available.

    """

    def __init__(self, decision_template: DecisionTemplate):
        param_names = []
        bounds = []
        defaults = []
        node_ids = []
        is_optimized = []

        for node_id, entry in decision_template.parameters.items():
            for name, numerical_param in entry.numerical.items():
                param_names.append(name)
                node_ids.append(node_id)
                bounds.append(numerical_param.bounds)
                defaults.append(numerical_param.default)
                is_optimized.append(numerical_param.is_optimized)

        self.param_names: Sequence[str] = param_names
        self.bounds: Sequence[tuple[float, float]] = bounds
        self.defaults: Sequence[number] = defaults
        self.node_ids: Sequence[int] = node_ids
        self.is_optimized: Sequence[bool] = is_optimized
        self.decision_templace: DecisionTemplate = decision_template

    def decision_from_array(self, decision_data: np.ndarray) -> Decision:
        """Transform a numpy array of selected parameters to the actual decision."""
        params: dict[int, dict[str, number]] = defaultdict(lambda: dict())
        iterator = zip(
            self.node_ids, self.param_names, self.is_optimized, self.defaults
        )
        i = 0
        for node_id, param_name, is_optimized, default in iterator:
            if is_optimized:
                params[node_id][param_name] = decision_data[i]
                i += 1
            else:
                params[node_id][param_name] = default
        return self.decision_templace.select_parameters(dict(params))

    def array_from_decision(self, decision: Decision) -> np.ndarray:
        """Transform the decision to a numpy array."""
        data = []
        iterator = zip(self.node_ids, self.param_names, self.is_optimized)

        for node_id, param_name, is_optimized in iterator:
            if is_optimized:
                data.append(decision.parameters[node_id][param_name])
        return np.array(data, dtype=float)

    def make_parameters_grid(self, num_samples: int = 20) -> np.ndarray:
        """Makes a grid of points to sample the parameters space."""

        # TODO: problems with log scale
        linspaces_1d = []
        for is_optimized, bounds in zip(self.is_optimized, self.bounds):
            if not is_optimized:
                continue
            linspaces_1d.append(np.linspace(*bounds, num=num_samples))

        x_space = np.atleast_2d(np.meshgrid(*linspaces_1d, indexing="ij"))
        x_space = x_space.reshape(x_space.shape[0], -1).T

        return x_space


class PerformancePredictor(ABC):
    @abstractmethod
    def select_solver_parameters(
        self,
        context: ProblemContext,
    ) -> PerformancePredictionData:
        """Choose to explore or to exploit and select a solver for the next time step."""

    @abstractmethod
    def _fit(self, full_contexts: np.ndarray, rewards: np.ndarray):
        pass

    def __init__(
        self, decision_template: DecisionTemplate, samples_before_fit: int = 10
    ) -> None:
        self.parameters_space: ParametersSpace = ParametersSpace(decision_template)
        self.decision_template: DecisionTemplate = decision_template
        self.x_space: np.ndarray = self.parameters_space.make_parameters_grid()
        self.memory_rewards: list[float] = []
        self.memory_contexts: list[np.ndarray] = []
        self.is_initialized: bool = False
        self.samples_before_fit: int = samples_before_fit

    def online_update(self, selection_data: SolverSelectionData) -> None:
        """Expects the data from one time step. `rewards` is a 1D array, size
        corresponds to the number of linear systems solver in this time step. `context`
        is a 1D array of the simulation characteristics this time step.

        """
        full_context, rewards = self._prepare_fit_data(selection_data)
        self._fit(full_context, rewards)

    def offline_update(self, selection_dataset: Sequence[SolverSelectionData]) -> None:
        """Expects the data from many time steps. `rewards` is a 1D array, size
        corresponds to the total number of linear systems. `context`
        is a 2D array: (num_rewards, num_features). You need to repeat the context for
        each linear system dyring one time step. The same applies to
        `numerical_parameters`.

        """
        full_contexts_list = []
        rewards_list = []
        for selection_data in selection_dataset:
            full_context, rewards = self._prepare_fit_data(selection_data)
            full_contexts_list.append(full_context)
            rewards_list.append(rewards)
        full_contexts = np.concatenate(full_contexts_list)
        rewards = np.concatenate(rewards_list)
        self._fit(full_contexts, rewards)

    def random_choice(self, context: ProblemContext) -> PerformancePredictionData:
        """Random exploration."""
        if self.x_space.size > 0:
            choice = np.random.choice(self.x_space.shape[0], size=1).item()
            parameters_chosen = self.x_space[choice]
        else:
            parameters_chosen = np.zeros(0)
        decision = self.parameters_space.decision_from_array(parameters_chosen)
        return PerformancePredictionData(
            score=DEFAULT_EXPECTATION, decision=decision, context=context
        )

    def _concatenate_context_parameters_space(self, context: np.ndarray):
        if len(self.x_space) == 0:
            return context.reshape(1, -1)
        numerical_actions = np.atleast_2d(self.x_space)
        context = np.broadcast_to(
            context, shape=(numerical_actions.shape[0], context.shape[0])
        )
        return np.concatenate([context, numerical_actions], axis=1)

    def _prepare_fit_data(self, selection_data: SolverSelectionData):
        rewards = np.array(selection_data.rewards)
        context = selection_data.prediction.context.get_array()
        decision = selection_data.prediction.decision
        solver_paramers = self.parameters_space.array_from_decision(decision)

        full_context = np.concatenate([context, solver_paramers])
        new_shape = (rewards.shape[0], full_context.shape[0])
        full_context = np.broadcast_to(full_context, shape=new_shape)
        return full_context, rewards


class PerformancePredictorEpsGreedy(PerformancePredictor):
    def __init__(
        self,
        decision_template: DecisionTemplate,
        samples_before_fit: int = 10,
        exploration: float = 0.5,
        exploration_rate: float = 0.9,
        regressor=None,
    ) -> None:
        self.exploration: float = exploration
        self.exploration_rate: float = exploration_rate

        if regressor is None:
            self.regressor = make_pipeline(
                StandardScaler(),
                # KNeighborsRegressor(n_neighbors=self.samples_before_fit),
                GradientBoostingRegressor(),
                # Ridge(),
            )
        else:
            self.regressor = regressor

        super().__init__(decision_template, samples_before_fit=samples_before_fit)

    def greedy_choice(self, context: ProblemContext) -> PerformancePredictionData:
        """Select optimal parameters based on the performance prediction."""
        full_contexts = self._concatenate_context_parameters_space(context.get_array())
        sample_prediction = self.regressor.predict(full_contexts)

        argmax = np.argmax(sample_prediction)
        expectation = sample_prediction[argmax]
        if self.x_space.size > 0:
            parameters_chosen = self.x_space[argmax]
        else:
            parameters_chosen = np.zeros(0)
        decision = self.parameters_space.decision_from_array(parameters_chosen)

        return PerformancePredictionData(
            score=float(expectation), decision=decision, context=context
        )

    def select_solver_parameters(
        self, context: ProblemContext
    ) -> PerformancePredictionData:
        greedy = self.is_initialized and (np.random.random() > self.exploration)

        if greedy:
            return self.greedy_choice(context)
        else:
            self.exploration *= self.exploration_rate
            return self.random_choice(context)

    def _fit(self, full_contexts: np.ndarray, rewards: np.ndarray):
        self.memory_contexts.extend(full_contexts.tolist())
        self.memory_rewards.extend(rewards.tolist())
        if len(self.memory_rewards) >= self.samples_before_fit:
            self.regressor.fit(self.memory_contexts, self.memory_rewards)
            self.is_initialized = True


class PerformancePredictorGaussianProcess(PerformancePredictor):
    def __init__(
        self,
        decision_template: DecisionTemplate,
        samples_before_fit: int = 10,
        alpha: float = 1e-1,
    ) -> None:
        self.alpha: float = alpha
        self.regressor: Pipeline
        super().__init__(
            decision_template=decision_template, samples_before_fit=samples_before_fit
        )

    def select_solver_parameters(
        self, context: ProblemContext
    ) -> PerformancePredictionData:
        if not self.is_initialized:
            return self.random_choice(context=context)
        full_contexts = self._concatenate_context_parameters_space(context.get_array())
        scaled = self.regressor.steps[0][1].transform(full_contexts)
        sample_prediction = self.regressor.steps[1][1].sample_y(scaled, n_samples=1)
        argmax = np.argmax(sample_prediction)
        expectation = sample_prediction[argmax]
        if self.x_space.size > 0:
            parameters_chosen = self.x_space[argmax]
        else:
            parameters_chosen = np.zeros(0)

        decision = self.parameters_space.decision_from_array(parameters_chosen)

        return PerformancePredictionData(
            score=float(expectation), decision=decision, context=context
        )

    def _fit(self, full_contexts: np.ndarray, rewards: np.ndarray):
        self.memory_contexts.extend(full_contexts.tolist())
        self.memory_rewards.extend(rewards.tolist())
        if len(self.memory_rewards) >= self.samples_before_fit:
            if not self.is_initialized:
                self.regressor = self._build_gp(full_contexts)
            self.regressor.fit(self.memory_contexts, self.memory_rewards)
            self.is_initialized = True

    def _build_gp(self, full_context: np.ndarray) -> Pipeline:
        feature_size = full_context.shape[1]
        ones = np.ones(feature_size)
        kernel = (
            kernels.RBF(length_scale=ones, length_scale_bounds=(1e0, 1e2))
            # kernels.ExpSineSquared()
            # kernels.RationalQuadratic()
            # + kernels.DotProduct()
            # + kernels.WhiteKernel(noise_level=1)
            # + 1
        )
        # RBF(5e-2, length_scale_bounds="fixed")
        return make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.alpha,
                n_restarts_optimizer=10,
                normalize_y=True,
            ),
        )


class PerformancePredictorRandom(PerformancePredictor):
    def select_solver_parameters(
        self, context: ProblemContext
    ) -> PerformancePredictionData:
        """Makes a random choice with a random score."""
        if self.x_space.size > 0:
            choice = np.random.choice(self.x_space.shape[0], size=1).item()
            parameters_chosen = self.x_space[choice]
        else:
            parameters_chosen = np.zeros(0)
        decision = self.parameters_space.decision_from_array(parameters_chosen)
        score = np.random.random() * DEFAULT_EXPECTATION
        return PerformancePredictionData(
            score=float(score), decision=decision, context=context
        )

    def _fit(self, full_contexts: np.ndarray, rewards: np.ndarray):
        pass


class OnlineStackingRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        base_regressors: Sequence[BaseEstimator],
        online_regressor: BaseEstimator,
    ):
        self.base_regressors: Sequence[BaseEstimator] = base_regressors
        self.online_regressor: BaseEstimator = online_regressor
        self.trust_model = Ridge()

    def _predict_regressors(self, X):
        predictions = []
        for reg in self.base_regressors:
            predictions.append(reg.predict(X))
        predictions.append(self.online_regressor.predict(X))
        return np.array(predictions).T

    def fit(self, X, y):
        self.online_regressor.fit(X, y)
        predictions = self._predict_regressors(X)
        self.trust_model.fit(predictions, y)
        return self

    def predict(self, X):
        predictions = self._predict_regressors(X)
        return self.trust_model.predict(predictions)


def make_performance_predictor(
    params: dict, solver_template: DecisionTemplate
) -> PerformancePredictor:
    predictor_name = params.get("predictor", "eps_greedy")
    samples_before_fit = params.get("samples_before_fit", 10)

    if predictor_name == "eps_greedy":
        print("Using epsilon-greedy exploration")
        exploration = params.get("exploration", 0.5)
        exploration_rate = params.get("exploration_rate", 0.9)
        regressor = params.get("regressor", "gradient_boosting")
        print("Using regressor:", regressor)

        if regressor == "gradient_boosting":
            predictor = PerformancePredictorEpsGreedy(
                decision_template=solver_template,
                exploration=exploration,
                exploration_rate=exploration_rate,
                samples_before_fit=samples_before_fit,
            )

        elif regressor == "mlp":
            predictor = PerformancePredictorEpsGreedy(
                decision_template=solver_template,
                exploration=exploration,
                exploration_rate=exploration_rate,
                samples_before_fit=samples_before_fit,
                regressor=make_pipeline(
                    StandardScaler(), MLPRegressor(hidden_layer_sizes=(100,))
                ),
            )

        elif regressor == "stacking":
            predictor = make_stacking(params=params, solver_template=solver_template)

    elif predictor_name == "gaussian_process":
        alpha = params.get("alpha", 1e-1)
        print("Using Gaussian process, exploration:", alpha)
        predictor = PerformancePredictorGaussianProcess(
            decision_template=solver_template,
            samples_before_fit=samples_before_fit,
            alpha=alpha,
        )

    elif predictor_name == "random":
        print("Using random exploration (it does not learn anything!)")
        predictor = PerformancePredictorRandom(
            decision_template=solver_template,
            samples_before_fit=samples_before_fit,
        )

    else:
        raise ValueError(predictor_name)

    return predictor


def make_stacking(
    params: dict, solver_template: DecisionTemplate
) -> PerformancePredictor:
    datasets_paths: Sequence[Sequence[str]] = params["stacking_datasets"]
    base_predictor_params: dict = params.get("base_predictor", {})

    offline_regressors = []
    for i, dataset_paths in enumerate(datasets_paths):
        print("Building base predictor", i, "with data:")
        for path in dataset_paths:
            print(path)
        data = load_data(dataset_paths)
        solver_data = []
        for entry in data:
            if entry.prediction.decision.subsolvers == solver_template.subsolvers:
                solver_data.append(entry)

        offline_predictor = make_performance_predictor(
            base_predictor_params, solver_template
        )
        offline_predictor.offline_update(solver_data)
        offline_regressors.append(offline_predictor.regressor)

    online_regressor = make_performance_predictor(
        base_predictor_params, solver_template
    ).regressor
    stacking = OnlineStackingRegressor(
        base_regressors=offline_regressors, online_regressor=online_regressor
    )

    samples_before_fit = params.get("samples_before_fit", 0)
    exploration = params.get("exploration", 0.5)
    exploration_rate = params.get("exploration_rate", 0.9)
    return PerformancePredictorEpsGreedy(
        decision_template=solver_template,
        samples_before_fit=samples_before_fit,
        exploration=exploration,
        exploration_rate=exploration_rate,
        regressor=stacking,
    )
