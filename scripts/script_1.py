from datetime import timedelta

import pandas as pd
from dateutil import parser
from tqdm import tqdm

import clients
import directories
import queries


start_date = "2024-11-07"
start_date = parser.parse(start_date)
end_date = "2024-11-10"
end_date = parser.parse(end_date)
current_date = start_date

dates = []
while current_date <= end_date:
    dates.append(current_date)
    current_date = current_date + timedelta(days=1)

for date in tqdm(dates, desc="Fetching queries"):
    date_str = date.strftime("%Y-%m-%d")
    query = queries.QUERY_USER_ITEMS.format(date=date_str)
    rows, columns = clients.trinox.fetch(query=query)

    df = pd.DataFrame(rows, columns=columns)
    df["pdid"].astype(str)
    df["pdid"] = df["pdid"].str.replace(r'\["?|"?\]', "", regex=True)

    df["uid"].astype(str)
    df.dropna(inplace=True)

    df = df.sort_values(by=["uid", "time"])
    df = df[["uid", "pdid", "time", "event"]]

    df = df[(df["uid"] != df["uid"].shift()) | (df["pdid"] != df["pdid"].shift())]

    workspace_path = directories.workspace("aboutpet", "item2vec", "v1")
    save_path = workspace_path.joinpath(f"user_items_{date_str}.csv").as_posix()
    df.to_csv(save_path, index=False)
