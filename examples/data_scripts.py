import os
import re
from typing import Sequence
from pathlib import Path


def append_experiment_name(path: str) -> Path:
    path: Path = Path(path).absolute()
    filename = path.name
    experiment_dir = path.parent / "performance"
    name = filename.removesuffix(".npy").removesuffix(".py")
    ids = ["-1"]
    for file in os.listdir(experiment_dir):
        if file.startswith(name):
            match = re.findall(r"(\d+).npy", file)
            ids.extend(match)
    ids = [int(x) for x in ids]
    max_id = max(ids) + 1

    experiment_dir.mkdir(exist_ok=True)
    return experiment_dir / f"{name}_{max_id}.npy"


def get_newest_data_paths(experiment_name: str, n_newest=3) -> Sequence[str]:
    name = experiment_name.removesuffix(".py").removesuffix(".npy")
    work_dir = Path(experiment_name).absolute().parent / "performance"
    data = {}
    for fname in os.listdir(work_dir):
        match = re.findall(rf"{name}_(\d+).npy", fname)
        if len(match) > 0:
            data[int(match[0])] = fname
    keys = sorted(data)[-n_newest:]
    values = [str(work_dir / data[key]) for key in keys]
    return values
