import pathlib

from item2vec.configs import settings


def delete_checkpoints():
    directory = pathlib.Path(settings.checkpoint_dirpath)
    for file in directory.glob("*"):
        if file.is_file():
            file.unlink()


def main():
    delete_checkpoints()


if __name__ == "__main__":
    main()
