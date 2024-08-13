import os

from pathlib import Path

project = Path(os.path.dirname(__file__)).parent

data = project.joinpath("data")
