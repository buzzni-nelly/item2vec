import glob
from pathlib import Path

import clients
import directories


def download_files(
    source_path: str,
    always: bool = False,
):

    filepath = directories.data.joinpath("user_items_*.pairs.jsonl")
    filepath = filepath.as_posix()
    pair_paths = list(map(Path, glob.glob(filepath)))

    if always or len(pair_paths) == 0:
        ceph = clients.ceph.CephClient()
        destination_path = directories.data.as_posix()
        ceph.download_dir(source_path, destination_path)
