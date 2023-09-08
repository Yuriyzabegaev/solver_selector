import os
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from solver_selector.data_structures import (
    NonlinearSolverStats,
    PerformancePredictionData,
    SolverSelectionData,
)


def make_solution_stats(
    perf: Sequence[SolverSelectionData], converged=True
) -> Sequence[NonlinearSolverStats]:
    solution_stats = np.array([data.nonlinear_solver_stats for data in perf])
    if not converged:
        return solution_stats
    converged_indices = make_converged_indices(perf)
    return solution_stats[converged_indices]


def make_converged_indices(perf: Sequence[SolverSelectionData]):
    solution_stats = make_solution_stats(perf, converged=False)
    return np.array([x.is_converged for x in solution_stats])


def make_solve_linear_system_time(perf, converged=True):
    solution_stats_converged = make_solution_stats(perf, converged=converged)
    return np.array(
        [
            y.solve_linear_system_time
            for x in solution_stats_converged
            for y in x.iterations
        ]
    )


def make_peclet_max(perf, converged=True):
    predictions = make_predictions(perf, converged=converged)
    return np.array([x.context.peclet_max for x in predictions])


def make_peclet_mean(perf, converged=True):
    predictions_converged = make_predictions(perf, converged=converged)
    return np.array([x.context.peclet_mean for x in predictions_converged])


def make_cfl_max(perf):
    predictions = make_predictions(perf)
    return np.array([x.context.cfl_max for x in predictions])


def make_cfl_mean(perf):
    predictions = make_predictions(perf)
    return np.array([x.context.cfl_mean for x in predictions])


def make_time_step(perf, converged=True):
    predictions = make_predictions(perf, converged=converged)
    return np.array([x.context.time_step_value for x in predictions])


def make_inlet_rate(perf, converged=True):
    predictions = make_predictions(perf, converged=converged)
    array = np.array([x.context.inlet_rate for x in predictions])
    return np.maximum(array, 1e-6)


def make_outlet_rate(perf, converged=True):
    predictions = make_predictions(perf, converged=converged)
    array = np.array([x.context.outlet_rate for x in predictions])
    return np.maximum(array, 1e-6)


def make_simulation_time(perf, converged=True):
    time_steps = make_time_step(perf, converged=converged)
    if not converged:
        conv = make_converged_indices(perf)
        time_steps[~conv] = 0
    return np.cumsum(time_steps)


def make_predictions(
    perf: Sequence[SolverSelectionData], converged=True
) -> Sequence[PerformancePredictionData]:
    predictions = np.array([data.prediction for data in perf])
    if converged:
        converged_indices = make_converged_indices(perf)
        predictions = predictions[converged_indices]
    return predictions


def sum_per_time_step(values, perf, converged=True):
    time_step_indices = make_time_step_numbers(perf, converged=converged)
    res = np.zeros(time_step_indices[-1] + 1)
    for i, val in enumerate(values):
        res[time_step_indices[i]] += val
    return res


def make_time_step_numbers(perf, converged=True):
    solution_stats = make_solution_stats(perf, converged=converged)
    return np.array([i for i, x in enumerate(solution_stats) for y in x.iterations])


def make_num_nonlinear_iters(perf, converged=True):
    solution_stats_converged = make_solution_stats(perf, converged=converged)
    return [x.num_nonlinear_iterations for x in solution_stats_converged]


def make_num_linear_iters(perf, converged=True):
    solution_stats_converged = make_solution_stats(perf, converged=converged)
    return np.array(
        [
            y.num_linear_iterations
            for x in solution_stats_converged
            for y in x.iterations
        ]
    )


def append_experiment_name(path: str | Path) -> Path:
    path = Path(path).absolute()
    filename = path.name
    experiment_dir = path.parent / "performance"
    experiment_dir.mkdir(exist_ok=True)
    name = filename.removesuffix(".npy").removesuffix(".py")
    ids = ["-1"]
    for file in os.listdir(experiment_dir):
        if file.startswith(name):
            match = re.findall(r"(\d+).npy", file)
            ids.extend(match)
    int_ids = [int(x) for x in ids]
    max_id = max(int_ids) + 1

    experiment_dir.mkdir(exist_ok=True)
    return experiment_dir / f"{name}_{max_id}.npy"


def get_newest_data_paths(experiment_name: str, n_newest=3) -> Sequence[str]:
    name = Path(experiment_name).name.removesuffix(".py").removesuffix(".npy")
    work_dir = Path(experiment_name).absolute().parent / "performance"
    data = {}
    for fname in os.listdir(work_dir):
        match = re.findall(rf"{name}_(\d+).npy", fname)
        if len(match) > 0:
            data[int(match[0])] = fname
    keys = sorted(data)[-n_newest:]
    values = [str(work_dir / data[key]) for key in keys]
    return values


def load_data(experiment_name, n_newest):
    paths = get_newest_data_paths(experiment_name, n_newest=n_newest)
    assert len(paths) == n_newest, "Data not found"
    print("Loading data:")
    for path in paths:
        print(path)
    return [np.load(path, allow_pickle=True).tolist() for path in paths]
