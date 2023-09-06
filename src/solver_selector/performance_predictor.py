from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Sequence
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct

from solver_selector.data_structures import (
    PerformancePredictionData,
    ProblemContext,
    SolverSelectionData,
)

from solver_selector.solver_space import (
    Decision,
    ParametersSpace,
    DecisionTemplate,
    number,
)

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
        self.is_optimized: tuple[bool] = is_optimized
        self.decision_templace: DecisionTemplate = decision_template

    def decision_from_array(self, decision_data: np.ndarray) -> Decision:
        """Transform a numpy array of selected parameters to the actual decision."""
        params = defaultdict(lambda: dict())
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

    def __init__(
        self, decision_template: DecisionTemplate, samples_before_fit: int = 10
    ) -> None:
        self.parameters_space: ParametersSpace = ParametersSpace(decision_template)
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

    def _fit(self, full_contexts: np.ndarray, rewards: np.ndarray):
        self.memory_contexts.extend(full_contexts.tolist())
        self.memory_rewards.extend(rewards.tolist())
        if len(self.memory_rewards) >= self.samples_before_fit:
            self.regressor.fit(self.memory_contexts, self.memory_rewards)
            self.is_initialized = True


class PerformancePredictorEpsGreedy(PerformancePredictor):
    def __init__(
        self,
        decision_template: DecisionTemplate,
        samples_before_fit: int = 10,
        exploration: float = 0.5,
        exploration_rate: float = 0.9,
    ) -> None:
        self.exploration: float = exploration
        self.exploration_rate: float = exploration_rate

        self.regressor = make_pipeline(
            StandardScaler(),
            # KNeighborsRegressor(n_neighbors=self.samples_before_fit),
            GradientBoostingRegressor(),
            # Ridge(),
        )

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


class PerformancePredictorGaussianProcess(PerformancePredictor):
    def __init__(
        self,
        decision_template: DecisionTemplate,
        samples_before_fit: int = 10,
    ) -> None:
        kernel = RBF() + DotProduct()
        # RBF(5e-2, length_scale_bounds="fixed")
        self.regressor = make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-1,
                n_restarts_optimizer=10,
                normalize_y=True,
            ),
        )
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
