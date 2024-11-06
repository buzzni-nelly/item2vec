import functools
import glob

import pandas as pd
from tqdm import tqdm

import directories

WINDOW_SIZE = 5


def build_graph(filepath: str):
    df = pd.read_csv("/Users/nelly/PycharmProjects/item2vec/assets/items.csv")
    items = df.set_index("pdid").T.to_dict()

    logs_df = pd.read_csv(filepath)

    logs_df.dropna(subset=["pdid"], inplace=True)
    logs_df["pdid"] = logs_df["pdid"].astype(str)

    def mapper(x, key=None):
        if key is None:
            raise ValueError
        return items[x][key] if x in items else None

    pid_mapper = functools.partial(mapper, key="pid")
    logs_df["pid"] = logs_df["pdid"].apply(pid_mapper)

    logs_df.dropna(subset=["pid"], inplace=True)
    logs_df["pid"] = logs_df["pid"].astype(int)

    category1_mapper = functools.partial(mapper, key="category1")
    category2_mapper = functools.partial(mapper, key="category2")
    category3_mapper = functools.partial(mapper, key="category3")
    logs_df["category1"] = logs_df["pdid"].apply(category1_mapper)
    logs_df["category2"] = logs_df["pdid"].apply(category2_mapper)
    logs_df["category3"] = logs_df["pdid"].apply(category3_mapper)

    logs_df = logs_df.sort_values(by=["uid", "time"])
    logs_df = logs_df[
        (logs_df["uid"] != logs_df["uid"].shift())
        | (logs_df["pid"] != logs_df["pid"].shift())
    ]

    logs_df = logs_df.reset_index(drop=True)

    edge_indices = []

    count = 0
    for row in logs_df.itertuples():
        if row.event != "purchase":
            continue

        current_uid = row.uid
        current_pid = row.pid
        current_category1 = row.category1
        current_category2 = row.category2
        current_index = row.Index  # Current index

        collected_pids = []

        for prev_idx in range(current_index - 1, -1, -1):
            prev_row = logs_df.iloc[prev_idx]
            if (
                prev_row.uid == current_uid
                and prev_row.category1 == current_category1
                and prev_row.category2 == current_category2
            ):
                collected_pids.append(prev_row.pid)
            else:
                break  # Stop if conditions do not match

        for pid in collected_pids:
            if pid != current_pid:
                edge_indices.append((pid, current_pid))
                count += 1

    print(f"Total edges created: {count}")
    return edge_indices


if __name__ == "__main__":
    path = directories.assets.joinpath("user_items_*.csv")
    filepaths = glob.glob(path.as_posix())
    filepaths.sort()
    filepaths = [x for x in filepaths if ".pairs." not in x]

    all_edge_indices = []

    for filepath in tqdm(filepaths[:3]):
        edge_index = build_graph(filepath)
        all_edge_indices.extend(edge_index)

    print(all_edge_indices)
