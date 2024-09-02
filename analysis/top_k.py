import pandas as pd
import torch

from item2vec import vocab
from item2vec.models import Item2VecModule

product_id_to_pid = vocab.load()
pid_to_product_id = {k: v for v, k in product_id_to_pid.items()}

model_path = "/Users/nelly/PycharmProjects/item2vec/checkpoints/epoch=15-step=1250000-train_loss=0.49.ckpt"
vocab_size = vocab.size()
item2vec_module = Item2VecModule.load_from_checkpoint(model_path, vocab_size=vocab_size)

item2vec_module.eval()
item2vec_module.freeze()

df = pd.read_csv("/Users/nelly/PycharmProjects/item2vec/csv/items.csv")

while True:
    pid = int(input("PID를 입력하세요: "))  # 사용자로부터 제품 ID 입력 받기

    show_columns = ["mall_product_name", "mall_product_category1", "click_count"]
    print(df[df["pid"] == pid][show_columns])

    # 임베딩 가져오기 및 cosine similarity 계산
    embeddings = item2vec_module.item2vec.embeddings.weight.data
    specific_embedding = embeddings[pid]
    products = torch.matmul(specific_embedding, embeddings.T)

    top_k = 10
    top_pids_products = torch.topk(products, top_k)
    pids_list_products = top_pids_products.indices.numpy().tolist()
    rows_products = df[df["pid"].isin(pids_list_products)]
    print("\n상위 dot product 제품 상세 정보:")
    print(rows_products[["mall_product_name", "mall_product_category1"]])

    normalized_embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    cosine_similarities = torch.matmul(
        normalized_embeddings[pid], normalized_embeddings.T
    )
    top_pids_cosine = torch.topk(cosine_similarities, top_k)
    pids_list_cosine = top_pids_cosine.indices.numpy().tolist()
    rows_cosine = df[df["pid"].isin(pids_list_cosine)]

    print("\n상위 cosine similarity 제품 상세 정보:")
    print(rows_cosine[["mall_product_name", "mall_product_category1"]])
