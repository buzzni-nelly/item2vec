import glob
import itertools
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import directories
from item2vec import vocab

WINDOW_SIZE = 5


def build_pairs_dataset(filepath: str):

    mapper = vocab.load()

    logs_df = pd.read_csv(filepath)

    logs_df.dropna(subset=["product_id"], inplace=True)
    logs_df["product_id"] = logs_df["product_id"].astype(str)

    logs_df["pid"] = logs_df["product_id"].map(mapper)

    logs_df.dropna(subset=["pid"], inplace=True)
    logs_df["pid"] = logs_df["pid"].astype(int)

    logs_df = logs_df.sort_values(by=["uid", "time"])
    logs_df = logs_df[["uid", "pid"]]

    # collect pairs by window size
    item_pairs = []
    for uid, group in tqdm(logs_df.groupby("uid")):
        pids = group["pid"].tolist()
        end = max(len(pids) - WINDOW_SIZE + 1, 1)
        chunks = [pids[x : x + WINDOW_SIZE] for x in range(end)]
        chunks = [list(dict.fromkeys(chunk)) for chunk in chunks]
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            a = len(chunk) // 2
            pairs = [(chunk[a], chunk[b]) for b in range(len(chunk)) if a != b]
            item_pairs.append(pairs)

    item_pairs = list(itertools.chain.from_iterable(item_pairs))

    # save jsonl
    pairs_df = pd.DataFrame(item_pairs, columns=["target", "positive"])

    # Save to CSV
    stem = Path(filepath).stem
    csv_path = directories.assets.joinpath(f"{stem}.pairs.csv")
    pairs_df.to_csv(csv_path, index=False)

    # remove data
    # os.remove(filepath)


if __name__ == "__main__":
    # log data paths
    path = directories.assets.joinpath("user_items_*.csv")
    filepaths = glob.glob(path.as_posix())
    filepaths.sort()

    from multiprocessing import Pool

    with Pool() as pool:
        pool.map(build_pairs_dataset, filepaths)
