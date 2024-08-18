import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("/Users/nelly/PycharmProjects/item2vec/csv/items.csv")

item_size = len(df)

df["log_click_count"] = np.log10(df["click_count"])

plt.figure(figsize=(10, 6))
plt.hist(
    df["log_click_count"], bins=50, color="blue", edgecolor="black"
)  # 적절한 빈 수 설정
plt.title(f"Histogram for {item_size} items")
plt.xlabel("Log 10 of Click Count")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
