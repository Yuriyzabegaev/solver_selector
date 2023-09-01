import os
import re
from pathlib import Path


def append_experiment_name(path: str) -> Path:
    path = Path(path).absolute()
    filename = path.name
    name = filename.removesuffix(".npy").removesuffix(".py")
    ids = ["-1"]
    for file in os.listdir():
        if file.startswith(name):
            match = re.findall(r"(\d+).npy", file)
            ids.extend(match)
    ids = [int(x) for x in ids]
    max_id = max(ids) + 1
    return path.parent / f"{name}_{max_id}.npy"
