import os
import pathlib

from pathlib import Path

project = Path(os.path.dirname(__file__)).parent

workspaces = project.joinpath("workspaces")


def workspace(site: str, model: str, version: str) -> pathlib.Path:
    return workspaces.joinpath(site, model, version)
