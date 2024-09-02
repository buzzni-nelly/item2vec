import os

from pathlib import Path

project = Path(os.path.dirname(__file__)).parent

assets = project.joinpath("assets")

pairs = assets.joinpath("user_items_*.pairs.csv")

item = assets.joinpath("items.json")
