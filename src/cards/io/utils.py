import json
from pathlib import Path


def read_json(file_path: Path | str) -> dict:
    with open(file_path, "r") as file:
        data = json.load(file)
    return data
