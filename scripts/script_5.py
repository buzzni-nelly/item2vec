import json
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from tqdm import tqdm

import clients
from item2vec import vocab
from item2vec.models import Item2VecModule


class BM25:

    def __init__(self, items: dict):
        args = [
            (x["pid"], x["pdid"], x["name"])
            for x in items.values()
            if x["name"] != "UNKNOWN"
        ]
        args.sort(key=lambda x: x[0])

        self.items = items
        self.pdids = [x[1] for x in args]
        self.names = [x[2] for x in args]
        self.tokenized_corpus = [re.split(r"[\s_,-/|\\]", x) for x in self.names]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def find(self, query):
        tokens = re.split(r"[\s_,-/|\\]", query)
        scores = self.bm25.get_scores(tokens)
        indices = np.argsort(scores)[::-1]

        items = []
        for idx in indices:
            score, pdid = scores[idx], self.pdids[idx]
            if int(score) == 0:
                continue
            items.append((pdid, score, self.items[pdid]))
        return items


def get_most_similar_active_item(pdids, items, limit: int = 30) -> str | None:
    for pdid in pdids:
        item = items[pdid]
        if item["clicked_count"] > limit:
            return pdid
    return None


def debug(
    target_item,
    top_k_items,
    top_k_scores,
    reranked_items,
    reranked_scores,
    show: bool = False,
):
    if not show:
        return
    print("")
    print(
        target_item["name"],
        target_item["category1"],
        target_item["category2"],
        target_item["purchase_count"],
    )
    print("=" * 10)
    count = 0
    for item, score in zip(top_k_items, top_k_scores):
        if target_item["category1"] != item["category1"]:
            continue
        print(
            item["name"],
            item["category1"],
            item["category2"],
            item["purchase_count"],
            score,
        )
        count += 1
        if count > 10:
            break
    print("**" * 10)
    count = 0
    for item, score in zip(reranked_items, reranked_scores):
        if target_item["category1"] != item["category1"]:
            continue
        print(
            item["name"],
            item["category1"],
            item["category2"],
            item["purchase_count"],
            round(float(score), 4),
        )
        count += 1
        if count > 10:
            break
    print("=" * 10)


def load_items():
    items_df = pd.read_csv("/Users/nelly/PycharmProjects/item2vec/assets/items.csv")
    items_dict = items_df.to_dict(orient="records")
    items_dict = {item["pdid"]: item for item in items_dict}
    return items_dict


def load_embeddings(embed_dim=64):
    model_path = "/tmp/checkpoints/last.ckpt"
    vocab_size = vocab.size()
    item2vec_module = Item2VecModule.load_from_checkpoint(
        model_path, vocab_size=vocab_size, embed_dim=embed_dim
    )
    item2vec_module.eval()
    item2vec_module.freeze()

    embeddings = item2vec_module.item2vec.embeddings.weight.data
    return embeddings


def main(embed_dim=64, click_count_limit: int = 20, candidate_k: int = 100):
    # Load vocabulary and model
    pdid2pid = vocab.load()
    pid2pdid = {v: k for k, v in pdid2pid.items()}

    embeddings = load_embeddings(embed_dim=embed_dim)
    items_dict = load_items()
    device = embeddings.device

    unknown_pids = [x["pid"] for x in items_dict.values() if x["name"] == "UNKNOWN"]
    embeddings[unknown_pids] = torch.zeros(embed_dim, device=device)

    recommendations = {}
    for i in tqdm(
        range(embeddings.shape[0]), desc="추천 점수를 계산 및 Redis 할당 중입니다.."
    ):
        if i in unknown_pids:
            continue

        pdid = pid2pdid[i]
        query_item = items_dict[pdid]
        category1, category2 = query_item["category1"], query_item["category2"]

        similarities = F.cosine_similarity(
            embeddings[i].unsqueeze(0), embeddings, dim=1
        )

        k = candidate_k
        top_k_values, top_k_indices = torch.topk(similarities, k)

        top_k_pdids = [pid2pdid[int(x)] for x in top_k_indices]
        top_k_items = [items_dict[pdid] for pdid in top_k_pdids]
        top_k_p_counts = [x["purchase_count"] for x in top_k_items]
        top_k_c_counts = [x["click_count"] for x in top_k_items]
        top_k_scores = top_k_values

        # 0.9 도 테스트 해볼 것
        category_scores = [
            1 if x["category2"] == category2 else 0.85 for x in top_k_items
        ]
        top_k_soft_values = F.softmax(top_k_values, dim=0)
        top_k_soft_p_values = F.softmax(
            torch.tensor(top_k_p_counts, device=device) / k, dim=0
        )
        top_k_soft_c_values = F.softmax(
            torch.tensor(top_k_c_counts, device=device) / (k**2), dim=0
        )

        combined_values = top_k_soft_values + top_k_soft_p_values + top_k_soft_c_values
        combined_values = combined_values * torch.tensor(category_scores, device=device)
        combined_values, merged_soft_indices = torch.sort(
            combined_values, descending=True
        )

        reranked_indices = top_k_indices[merged_soft_indices]
        reranked_k_pdids = [pid2pdid[int(x)] for x in reranked_indices]
        reranked_items = [items_dict[pdid] for pdid in reranked_k_pdids]
        reranked_scores = combined_values

        debug(
            query_item,
            top_k_items,
            top_k_scores,
            reranked_items,
            reranked_scores,
            show=True,
        )

        candidate_scores = [
            {"pdid": x["pdid"], "score": round(float(s), 4)}
            for x, s in zip(reranked_items, reranked_scores)
            if category1 == x["category1"]
        ]

        recommendations[pdid] = candidate_scores

    # bm25 = BM25(items_dict)
    # for i in tqdm(range(embeddings.shape[0]), desc="낙오 된 아이템 처리를 진행합니다.."):
    #     if i in unknown_pids:
    #         continue
    #
    #     pdid = pid2pdid[i]
    #     query_item = items_dict[pdid]
    #     name = query_item["name"]
    #     category1, category2 = query_item["category1"], query_item["category2"]
    #     click_count = query_item["click_count"]
    #
    #     if click_count >= click_count_limit:
    #         continue
    #
    #     candidate_scores = bm25.find(name)
    #     valid_pdids = [
    #         c_pdid
    #         for c_pdid, c_score, c_item in candidate_scores
    #         if c_pdid != pdid
    #         and c_pdid in recommendations
    #         and c_item["category1"] == category1
    #         and c_item["category2"] == category2
    #         and c_item["purchase_count"] >= click_count_limit
    #         and c_item["name"] != "UNKNOWN"
    #     ]
    #     if valid_pdids:
    #         c_pdid = next(iter(valid_pdids))
    #         recommendations[pdid] = recommendations[c_pdid]

    pipeline = clients.redis.aiaas_6.pipeline()
    for k, v in recommendations.items():
        key = f"i2i:aboutpet:i2v:v1:{k}"
        pipeline.set(key, json.dumps(v))
        pipeline.expire(key, 30 * 24 * 60 * 60)
    pipeline.execute()

    print(len(recommendations))


if __name__ == "__main__":
    main()
