import json
from collections import defaultdict, Counter
from datetime import datetime, timedelta

import pandas as pd

import clients
import queries


def fetch_click_purchase_footsteps(begin_date: datetime) -> list:
    begin_date = begin_date.strftime("%Y-%m-%d")
    query = queries.ABOUTPET_CLICK_PURCHASE_FOOTSTEP.format(date=begin_date)
    rows, columns = clients.trinox.fetch(query)
    df = pd.DataFrame(rows, columns=columns)
    df = df.dropna(subset=["pid", "target_pid"])
    df = df[df["purchase_time"].notna()]

    df["source_pdid"] = df["pid"].apply(lambda x: f"aboutpet_{x}")
    df["target_pdid"] = df["target_pid"].apply(lambda x: f"aboutpet_{x}")
    df = df[["source_pdid", "target_pdid"]]
    df = df[df["source_pdid"] != df["target_pdid"]]
    return df.values.tolist()


def fetch_click_click_footsteps(begin_date: datetime) -> list:
    begin_date = begin_date.strftime("%Y-%m-%d")
    query = queries.ABOUTPET_CLICK_CLICK_FOOTSTEP.format(date=begin_date)
    rows, columns = clients.trinox.fetch(query)
    df = pd.DataFrame(rows, columns=columns)
    df = df.dropna(subset=["pid", "target_pid"])

    df["source_pdid"] = df["pid"].apply(lambda x: f"aboutpet_{x}")
    df["target_pdid"] = df["target_pid"].apply(lambda x: f"aboutpet_{x}")
    df = df[["source_pdid", "target_pdid"]]
    df = df[df["source_pdid"] != df["target_pdid"]]
    return df.values.tolist()

def main():
    pipeline = clients.redis.aiaas_6.pipeline()

    begin_date = datetime.now() - timedelta(days=4)
    print("Fetching click-purchase footsteps from trino..")
    click_purchase_footsteps = fetch_click_purchase_footsteps(begin_date=begin_date)
    print("Fetching click-click footsteps from trino..")
    click_click_footsteps = fetch_click_click_footsteps(begin_date=begin_date)

    aggregated_click_purchase = defaultdict(Counter)
    for source_pdid, target_pdid in click_purchase_footsteps:
        aggregated_click_purchase[source_pdid][target_pdid] += 1

    aggregated_click_click = defaultdict(Counter)
    for source_pdid, target_pdid in click_click_footsteps:
        aggregated_click_click[source_pdid][target_pdid] += 1

    for source_pdid, counter in aggregated_click_purchase.items():
        key = f"i2i:aboutpet:pastbias:p1:{source_pdid}"
        target_scores = [{"pdid": x, "score": c} for x, c in counter.most_common()]
        pipeline.set(key, json.dumps(target_scores))
        pipeline.expire(key, 3 * 24 * 60 * 60)

    for source_pdid, counter in aggregated_click_click.items():
        key = f"i2i:aboutpet:pastbias:c1:{source_pdid}"
        target_scores = [{"pdid": x, "score": c} for x, c in counter.most_common()]
        pipeline.set(key, json.dumps(target_scores))
        pipeline.expire(key, 3 * 24 * 60 * 60)

    pipeline.execute()
    print(f"Total {len(aggregated_click_purchase)} click-purchase footsteps updated on redis")
    print(f"Total {len(aggregated_click_click)} click-click footsteps updated on redis")


if __name__ == "__main__":
    main()
