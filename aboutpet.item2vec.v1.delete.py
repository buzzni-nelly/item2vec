import directories
from item2vec.configs import Settings as Item2VecSettings
from item2vec.volume import Volume

config_path = directories.config("aboutpet", "item2vec", "v1")
settings = Item2VecSettings.load(config_path)


def delete_checkpoints():
    volume = Volume(company_id="aboutpet", model="item2vec", version="v1")
    print(f"Removing checkpoints from {volume.checkpoints_dirpath}...")
    for file in volume.checkpoints_dirpath.glob("*"):
        if file.is_file():
            file.unlink()


def main():
    delete_checkpoints()


if __name__ == "__main__":
    main()
