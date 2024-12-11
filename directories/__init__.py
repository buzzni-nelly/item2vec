import os
import pathlib

from pathlib import Path

project = Path(os.path.dirname(__file__)).parent

configs = project.joinpath("configs")

workspaces = project.joinpath("workspaces")


def workspace(company_id: str, model: str, version: str) -> pathlib.Path:
    return workspaces.joinpath(company_id, model, version)


def config(company_id: str, model: str, version: str) -> pathlib.Path:
    return configs.joinpath(f"{company_id}.{model}.{version}.yaml")
