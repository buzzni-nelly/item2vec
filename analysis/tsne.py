import pandas as pd

import torch
from sklearn.manifold import TSNE
import plotly.express as px

import directories
from item2vec import vocab
from item2vec.models import Item2VecModule

# vocab 모듈에서 사전 로드
mapper = vocab.load()

# Item2Vec 모델 로드
# weight decay: 0.001
# dim_embed: 256
model_path = "/tmp/checkpoints/last.ckpt"
vocab_size = vocab.size()
item2vec_module = Item2VecModule.load_from_checkpoint(
    model_path, vocab_size=vocab_size, embed_dim=64
)

item2vec_module.eval()
item2vec_module.freeze()

# 임베딩 가져오기
embeddings = item2vec_module.item2vec.embeddings.weight.data

# 아이템 데이터 로드
items_path = directories.assets.joinpath("items.csv").as_posix()
df = pd.read_csv(items_path)
df = df[df["category1"] != "UNKNOWN"]
df = df.sort_values(by="click_count", ascending=False)

# 상위 1000개의 pid 추출 및 해당 정보 포함
top_pids = df.head(10000)

# PyTorch 텐서로 변환
indices = torch.LongTensor(top_pids["pid"].tolist())
selected_embeddings = embeddings[indices]

# 임베딩을 Numpy 배열로 변환
features = selected_embeddings.cpu().numpy()

# t-SNE를 사용하여 2차원으로 축소
tsne = TSNE(
    n_components=2,
    random_state=0,
    perplexity=100,
    early_exaggeration=50,
    metric="cosine",
)
projections = tsne.fit_transform(features)

# 시각화를 위해 결과 데이터 프레임 생성
result_df = pd.DataFrame(projections, columns=["x", "y"])
result_df["name"] = top_pids["name"].values
result_df["category1"] = top_pids["category1"].values
result_df["category2"] = top_pids["category2"].values
result_df["click_count"] = top_pids["click_count"].values

# 시각화
fig = px.scatter(
    result_df,
    x="x",
    y="y",
    color="category2",
    hover_data=["name", "category1", "category2", "click_count"],
)
fig.update_traces(marker_size=8)
fig.show()