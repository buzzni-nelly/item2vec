import pathlib

import directories
from item2vec.configs import Settings

settings = Settings.load(directories.config("aboutpet", "item2vec", "v1"))


def delete_checkpoints():
    directory = pathlib.Path(settings.checkpoint_dirpath)
    for file in directory.glob("*"):
        if file.is_file():
            file.unlink()


def main():
    delete_checkpoints()


if __name__ == "__main__":
    main()
