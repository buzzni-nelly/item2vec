import os

from pathlib import Path

project = Path(os.path.dirname(__file__)).parent

csv = project.joinpath("csv")

pairs = csv.joinpath("user_items_2024-07-01.pairs.csv")

item = csv.joinpath("items.json")
