import os

import pandas as pd
import plotly.express as px
import torch
from sklearn.manifold import TSNE

from lightgcn.models import LightGCNModel
from lightgcn.volume import Volume

model_path = "/tmp/checkpoints/last.ckpt"
volume = Volume("aboutpet", "lightgcn", "v1")

edge_index = os.path.join(volume.workspace_path, 'edge.click.indices.csv')
edge_df = pd.read_csv(edge_index)
sources = edge_df["source"].values
targets = edge_df["target"].values
edge_index = torch.tensor([sources, targets], dtype=torch.long)

lightgcn_module = LightGCNModel.load_from_checkpoint(
    model_path,
    num_users=volume.count_users(),
    num_items=volume.count_items(),
    embedding_dim=64,
    num_layers=3,
    edge_index=edge_index,
)

lightgcn_module.setup()
lightgcn_module.eval()
lightgcn_module.freeze()

# 임베딩 가져오기
user_embeddings, item_embeddings = lightgcn_module.forward()

# 아이템 데이터 로드
selected_embeddings = item_embeddings.weights

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

# 시각화
fig = px.scatter(
    result_df,
    x="x",
    y="y",
)
fig.update_traces(marker_size=8)
fig.show()
