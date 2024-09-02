from datetime import timedelta

import pandas as pd
from dateutil import parser
from tqdm import tqdm

import clients
import directories
import queries


start_date = "2024-07-01"
start_date = parser.parse(start_date)
end_date = "2024-08-31"
end_date = parser.parse(end_date)
current_date = start_date

dates = []
while current_date <= end_date:
    dates.append(current_date)
    current_date = current_date + timedelta(days=1)

for date in tqdm(dates, desc="Fetching queries"):
    date_str = date.strftime("%Y-%m-%d")
    query = queries.query_user_items.format(date=date_str)
    rows, columns = clients.trinox.fetch(query=query)

    data = pd.DataFrame(rows, columns=columns)
    data["product_id"].astype(str)
    save_path = directories.assets.joinpath(f"user_items_{date}.csv").as_posix()
    data.to_csv(save_path, index=False)
