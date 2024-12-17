import directories
from item2vec.configs import Settings as Item2VecSettings
from item2vec.volume import Volume

config_path = directories.config("aboutpet", "item2vec", "v1")
settings = Item2VecSettings.load(config_path)


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
