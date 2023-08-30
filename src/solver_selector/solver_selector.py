from typing import Sequence

import numpy as np

from solver_selector.solver_space import Decision, SolverConfigNode, DecisionTemplate
from solver_selector.performance_predictor import PerformancePredictor
from solver_selector.data_structures import (
    NonlinearIterationStats,
    NonlinearSolverStats,
    SolverSelectionData,
    ProblemContext,
    PerformancePredictionData,
)


class RewardPicker:
    def __init__(self):
        self._worst_work_time: float = 100.0

    def pick_rewards(self, solution_data: NonlinearSolverStats):
        results = [self.pick_reward(item) for item in solution_data.iterations]
        return np.array(results)

    def pick_reward(self, linear_data: NonlinearIterationStats):
        if linear_data.linear_solver_converged:
            time = (
                linear_data.solve_linear_system_time
                # + linear_data.update_preconditioner_time
                # + linear_data.assembly_time
            )
            time = max(time, 1e-2)
            self._worst_work_time = max(self._worst_work_time, time)
        else:
            time = self._worst_work_time * 2
        reward = -np.log(time)
        return reward

    @staticmethod
    def reverse_transform(data: np.ndarray):
        return np.exp(-data)


class SolverSelector:
    """Selects the optimal solver based on statistical data. Exploits/explores."""

    def __init__(
        self,
        solver_space: SolverConfigNode,
        predictors: Sequence[PerformancePredictor],
    ):
        self.solver_space: SolverConfigNode = solver_space
        self.all_solvers: Sequence[
            DecisionTemplate
        ] = self.solver_space.get_all_solvers()
        self.memory: list[SolverSelectionData] = []
        self.predictors: Sequence[PerformancePredictor] = predictors

    def show_all_solvers(self) -> None:
        for i, solver_template in enumerate(self.all_solvers):
            default_solver = solver_template.use_defaults()
            config = self.solver_space.config_from_decision(default_solver)
            print(f"{i}:", self.solver_space.format_config(config))

    def select_solver(
        self,
        context: ProblemContext,
    ) -> PerformancePredictionData:
        """Select a solver for the given simulation context based on previously observed
        data. Returns a decision which should be converted into a proper configuration
        with the method `SolverConfigNode.config_from_decision`.

        """
        # Each predictor selects best parameters of the corresponding solver and its
        # score.
        predictions: list[PerformancePredictionData] = []
        for predictor in self.predictors:
            # need a copy because we'll set numerical decision values
            # decision = copy.deepcopy(decision, {})
            predictions.append(predictor.select_solver_parameters(context))

        # Select the proposed algorithm with maximum score.
        arm_id = np.argmax([float(x.score) for x in predictions])
        prediction = predictions[arm_id]
        return prediction

    def learn_performance_online(self, selection_data: SolverSelectionData) -> None:
        """Update the corresponing predictor with new performance data."""
        predictor_idx = self._get_solver_idx(selection_data.prediction.decision)
        self.predictors[predictor_idx].online_update(selection_data)
        self.memory.append(selection_data)

    def learn_performance_offline(
        self, selection_dataset: Sequence[SolverSelectionData]
    ) -> None:
        """Update the corresponing predictor with the performance data from previous
        runs. Used for warm-starting.

        """
        datasets_for_predictors = [[] for _ in range(len(self.predictors))]
        for selection_data in selection_dataset:
            decision_idx = self._get_solver_idx(selection_data.prediction.decision)
            datasets_for_predictors[decision_idx].append(selection_data)

        for dataset, predictor in zip(datasets_for_predictors, self.predictors):
            predictor.offline_update(dataset)

    def _get_solver_idx(self, decision: Decision) -> int:
        for idx, solver in enumerate(self.all_solvers):
            if solver.subsolvers == decision.subsolvers:
                return idx
        raise ValueError("Index not found.")
