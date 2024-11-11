import glob
import json

import pandas as pd

import clients
import directories


def load_and_concat_csvs(filepaths: list[str]):
    dtype = {"uid": str, "pdid": str, "time": float, "event": str}
    dfs = [pd.read_csv(filepath, dtype=dtype) for filepath in filepaths]
    concatenated_df = pd.concat(dfs, ignore_index=True)
    return concatenated_df


def aggregate_event_counts(df: pd.DataFrame):
    click_counts_df = df.groupby("pdid").size().reset_index(name="click_count")
    purchase_counts_df = (
        df[df["event"] == "purchase"]
        .groupby("pdid")
        .size()
        .reset_index(name="purchase_count")
    )
    event_counts_df = pd.merge(
        click_counts_df, purchase_counts_df, on="pdid", how="outer"
    ).fillna(0)
    return event_counts_df


def fetch_product_details(pdids):
    # Retrieve product details from MongoDB
    products = clients.mongo.p32712.list_products(pdids)
    product_data = {
        product["pdid"]: {
            "name": product.get("name", ""),
            "category1": product.get("category1", ""),
            "category2": product.get("category2", ""),
            "category3": product.get("category3", ""),
        }
        for product in products
    }
    return product_data


def add_product_details(df: pd.DataFrame):
    pdids = df["pdid"].unique().tolist()
    products = clients.mongo.p32712.list_products(pdids)
    products_dict = {
        x["_id"]: {
            "name": x.get("name"),
            "category1": x.get("category1"),
            "category2": x.get("category2"),
            "category3": x.get("category3"),
        }
        for x in products
    }
    df["name"] = df["pdid"].map(
        lambda x: products_dict.get(x, {}).get("name", "UNKNOWN")
    )
    df["category1"] = df["pdid"].map(
        lambda x: products_dict.get(x, {}).get("category1", "UNKNOWN")
    )
    df["category2"] = df["pdid"].map(
        lambda x: products_dict.get(x, {}).get("category2", "UNKNOWN")
    )
    df["category3"] = df["pdid"].map(
        lambda x: products_dict.get(x, {}).get("category3", "UNKNOWN")
    )
    return df


if __name__ == "__main__":
    workspace_path = directories.workspace("aboutpet", "item2vec", "v1")
    path_pattern = workspace_path.joinpath("user_items_*.csv").as_posix()
    filepaths = glob.glob(path_pattern)
    filepaths = [x for x in filepaths if ".pairs." not in x]
    filepaths.sort()

    # TODO: concat 알고리즘 update 필요 for 대용량 like gsshop
    concatenated_df = load_and_concat_csvs(filepaths)
    items_df = aggregate_event_counts(concatenated_df)

    # Add product details (name, category) from MongoDB
    items_df = add_product_details(items_df)

    items_df = items_df.sort_values(by="pdid").reset_index(drop=True)
    # event_counts_df = event_counts_df[event_counts_df['click_count'] >= 10]
    items_df = items_df[items_df["name"] != "UNKNOWN"]
    items_df["pid"] = pd.factorize(items_df["pdid"])[0]

    items_df = items_df[
        [
            "pid",
            "pdid",
            "click_count",
            "purchase_count",
            "name",
            "category1",
            "category2",
            "category3",
        ]
    ]
    items_df.to_csv(
        workspace_path.joinpath("items.csv").as_posix(),
        index=False,
        header=True,
    )

    pdid_to_pid = items_df.set_index("pdid")["pid"].to_dict()
    with open(workspace_path.joinpath("items.json").as_posix(), "w") as f:
        json.dump(pdid_to_pid, f)
