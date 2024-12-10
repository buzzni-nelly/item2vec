import os

import pandas as pd
import plotly.express as px
import torch
from sklearn.manifold import TSNE

from item2vec.models import GraphBPRItem2Vec
from item2vec.volume import Volume

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_path = "/tmp/checkpoints/last.ckpt"
volume = Volume("aboutpet", "item2vec", "v1")

item2vec_module = GraphBPRItem2Vec.load_from_checkpoint(
    model_path,
    vocab_size=volume.vocab_size(),
    purchase_edge_index_path=volume.workspace_path.joinpath("edge.purchase.indices.csv"),
    embed_dim=128,
)

item2vec_module.setup()
item2vec_module.eval()
item2vec_module.freeze()
item2vec_module.to("cpu")

# 임베딩 가져오기
embeddings = item2vec_module.get_graph_embeddings(num_layers=3)

# 아이템 데이터 로드
items = volume.items().values()
df = pd.DataFrame(items)
df = df[df["category1"] != "UNKNOWN"]
df = df.sort_values(by="click_count", ascending=False)

# 상위 1000개의 pid 추출 및 해당 정보 포함
top_pids = df.head(10000)

# PyTorch 텐서로 변환
indices = torch.LongTensor(top_pids["pidx"].tolist())
selected_embeddings = embeddings[indices]

# 임베딩을 Numpy 배열로 변환
features = selected_embeddings.cpu().numpy()

# t-SNE를 사용하여 2차원으로 축소
tsne = TSNE(
    n_components=2,
    random_state=0,
    perplexity=30,
    early_exaggeration=12.0,
    metric="cosine",
)
projections = tsne.fit_transform(features)

# 시각화를 위해 결과 데이터 프레임 생성
result_df = pd.DataFrame(projections, columns=["x", "y"])
result_df["name"] = top_pids["name"].values
result_df["category1"] = top_pids["category1"].values
result_df["category2"] = top_pids["category2"].values
result_df["category3"] = top_pids["category3"].values
result_df["click_count"] = top_pids["click_count"].values

# 시각화
fig = px.scatter(
    result_df,
    x="x",
    y="y",
    color="category2",
    hover_data=["name", "category1", "category2", "category3", "click_count"],
)
fig.update_traces(marker_size=8)
fig.show()
