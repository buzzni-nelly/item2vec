import pandas as pd
import torch

from item2vec import vocab
from item2vec.models import Item2VecModule
import plotly.express as px

df = pd.read_csv("/Users/nelly/PycharmProjects/item2vec/csv/items.csv")

df = df.sort_values(by="click_count", ascending=False).head(20000)

model_path = "/Users/nelly/PycharmProjects/item2vec/checkpoints/v2-epoch=5-step=470000-train_loss=0.50.ckpt"
vocab_size = vocab.size()
item2vec_module = Item2VecModule.load_from_checkpoint(
    model_path,
    vocab_size=vocab_size,
    embed_dim=128,
)
item2vec_module.eval()
item2vec_module.freeze()

embeddings = item2vec_module.item2vec.embeddings.weight.data
norms = torch.norm(embeddings, dim=1)

df["norm"] = df["pid"].apply(lambda x: norms[x].item() if x < len(norms) else None)

fig = px.scatter(
    df,
    x="click_count",
    y="norm",
    labels={"norm": "Norm of Embeddings", "click_count": "Click Count"},
    hover_data=["pid", "mall_product_name", "click_count", "norm"],
    title="Top 20,000 Click Count vs Norm of Embeddings",
)
fig.update_layout(title="Top 20,000 Click Count vs Norm of Embeddings")
fig.show()
