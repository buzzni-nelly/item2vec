import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px

import directories
from item2vec import vocab
from item2vec.models import Item2VecModule

# Load vocabulary from the vocab module
mapper = vocab.load()

# Load Item2Vec model
model_path = "/Users/nelly/PycharmProjects/item2vec/checkpoints/v2-epoch=5-step=470000-train_loss=0.50.ckpt"
vocab_size = vocab.size()
item2vec_module = Item2VecModule.load_from_checkpoint(
    model_path, vocab_size=vocab_size, embed_dim=128
)

item2vec_module.eval()
item2vec_module.freeze()

embeddings = item2vec_module.item2vec.embeddings.weight.data

items_path = directories.csv.joinpath("items.csv").as_posix()
df = pd.read_csv(items_path)
df = df.sort_values(by="click_count", ascending=False)

top_pids = df.head(20000)

indices = torch.LongTensor(top_pids["pid"].tolist())
selected_embeddings = embeddings[indices]

features = selected_embeddings.numpy()

tsne = TSNE(
    n_components=3,
    random_state=0,
    perplexity=100,
    early_exaggeration=50,
    metric="cosine",
)
projections = tsne.fit_transform(features)

kmeans = KMeans(n_clusters=40, random_state=0).fit(projections)

result_df = pd.DataFrame(projections, columns=["x", "y", "z"])
result_df["mall_product_name"] = top_pids["mall_product_name"].values
result_df["mall_product_category1"] = top_pids["mall_product_category1"].values
result_df["click_count"] = top_pids["click_count"].values
result_df["cluster"] = kmeans.labels_
result_df["pid"] = top_pids["pid"].values

fig = px.scatter_3d(
    result_df,
    x="x",
    y="y",
    z="z",
    color="cluster",
    hover_data=["mall_product_name", "mall_product_category1", "click_count", "pid"],
    title="3D K-means Clustering of Item Embeddings",
    labels={"cluster": "Cluster"},
    color_continuous_scale=px.colors.sequential.Viridis
    + px.colors.sequential.Pinkyl[2:],
)
fig.update_traces(marker_size=3)
fig.show()
