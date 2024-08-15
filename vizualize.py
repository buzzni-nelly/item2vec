import torch
from sklearn.manifold import TSNE
import plotly.express as px

from item2vec import vocab
from item2vec.models import Item2VecModule

mapper = vocab.load()

model_path = (
    "/Users/nelly/PycharmProjects/item2vec/checkpoints/epoch=1-step=225876.ckpt"
)
vocab_size = vocab.size()
item2vec_module = Item2VecModule.load_from_checkpoint(model_path, vocab_size=vocab_size)

item2vec_module.eval()
item2vec_module.freeze()

embeddings = item2vec_module.item2vec.embeddings.weight.data

indices = torch.randperm(len(embeddings))[:100000]
selected_embeddings = embeddings[indices]

features = selected_embeddings.numpy()

tsne = TSNE(n_components=2, random_state=0, perplexity=50, early_exaggeration=20)
projections = tsne.fit_transform(features)

fig = px.scatter(projections, x=0, y=1)
fig.update_traces(marker_size=8)
fig.show()
