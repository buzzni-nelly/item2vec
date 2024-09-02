import json

import pandas as pd

import clients.trinox
import directories
import queries

START_DATE = "2024-07-01"
END_DATE = "2024-08-15"

print(f"Fetching query for {START_DATE} to {END_DATE}")
query = queries.query_joined_items.format(start_date=START_DATE, end_date=END_DATE)
rows, columns = clients.trinox.fetch(query=query)

items_df = pd.DataFrame(rows, columns=columns)
items_df = items_df.dropna()
items_df["product_id"] = items_df["product_id"].astype(str)
items_df["pid"] = items_df["pid"].astype(int)

print("Saving..")
save_path = directories.assets.joinpath("items.csv").as_posix()
items_df.to_csv(save_path, index=False)

with open(directories.assets.joinpath("items.json").as_posix(), "w") as f:
    json.dump(dict(zip(items_df["product_id"], items_df["pid"])), f)

print(f"Saved total: {len(items_df)}")
