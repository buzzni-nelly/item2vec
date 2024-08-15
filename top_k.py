import torch

from item2vec import vocab
from item2vec.models import Item2VecModule

product_id_to_pid = vocab.load()
pid_to_product_id = {k: v for v, k in product_id_to_pid.items()}


model_path = (
    "/Users/nelly/PycharmProjects/item2vec/checkpoints/epoch=1-step=225876.ckpt"
)
vocab_size = vocab.size()
item2vec_module = Item2VecModule.load_from_checkpoint(model_path, vocab_size=vocab_size)

item2vec_module.eval()
item2vec_module.freeze()

while True:
    product_id = input("Enter product id: ")
    pid = product_id_to_pid[f"{product_id}"]

    embeddings = item2vec_module.item2vec.embeddings.weight.data
    specific_embedding = embeddings[pid]

    similarities = torch.matmul(specific_embedding, embeddings.T)

    top_10_indices = torch.topk(similarities, 10).indices

    top_k_product_ids = [pid_to_product_id[pid]] + [
        pid_to_product_id[int(idx)] for idx in top_10_indices
    ]

    print(top_k_product_ids)
