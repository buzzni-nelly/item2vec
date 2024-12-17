import directories
from item2vec.configs import Settings
from item2vec.volume import Volume

settings = Settings.load(directories.config("aboutpet", "item2vec", "v1"))


def delete_checkpoints():
    volume = Volume(company_id="aboutpet", model="item2vec", version="v1")
    item2vec_checkpoint_dir_path = volume.workspace_path.joinpath("checkpoints")
    print(f"Removing checkpoints from {item2vec_checkpoint_dir_path}...")
    for file in item2vec_checkpoint_dir_path.glob("*"):
        if file.is_file():
            file.unlink()


def main():
    delete_checkpoints()


if __name__ == "__main__":
    main()
