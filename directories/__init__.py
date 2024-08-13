import os

from pathlib import Path

project = Path(os.path.dirname(__file__)).parent

data = project.joinpath("data")

pairs = data.joinpath("user_items_*.pairs.jsonl")

item = data.joinpath("items.json")
