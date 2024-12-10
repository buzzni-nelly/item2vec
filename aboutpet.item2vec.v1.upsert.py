import collections
import json

import torch
from retry import retry
from tqdm import tqdm

import clients
import directories
from item2vec.configs import Settings
from item2vec.models import GraphBPRItem2Vec
from item2vec.volume import Volume

RELEASE = "i2i"

COMPANY_ID = "aboutpet"

MODEL_NAME = "i2v"

VERSION = "v1"

settings = Settings.load(directories.config(COMPANY_ID, MODEL_NAME, VERSION))


def debug(
    target_item,
    top_k_items,
    top_k_scores,
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

    print("=" * 20)
    for item, score in zip(top_k_items[:10], top_k_scores[:10]):
        if target_item["category1"] != item["category1"]:
            continue
        print(
            item["name"],
            item["category1"],
            item["category2"],
            item["purchase_count"],
            round(float(score), 4),
        )
    print("=" * 20)


def load_embeddings(volume: Volume, embed_dim: int = 256):
    model_path = f"{settings.checkpoint_dirpath}/last.ckpt"
    vocab_size = volume.vocab_size()
    item2vec_module = GraphBPRItem2Vec.load_from_checkpoint(
        model_path,
        vocab_size=vocab_size,
        purchase_edge_index_path=volume.workspace_path.joinpath("edge.purchase.indices.csv"),
        embed_dim=embed_dim,
    )
    item2vec_module.setup()
    item2vec_module.eval()
    item2vec_module.freeze()

    embeddings = item2vec_module.get_graph_embeddings(num_layers=3)
    return embeddings


def upload(aggregated_scores: dict):
    pipeline = clients.redis.aiaas_6.pipeline()
    for pdid, scores in aggregated_scores.items():
        key = f"{RELEASE}:{COMPANY_ID}:{MODEL_NAME}:{VERSION}:{pdid}"
        pipeline.set(key, json.dumps(scores))
        pipeline.expire(key, 30 * 24 * 60 * 60)
    pipeline.execute()
    print(f"Total {len(aggregated_scores)} items updated on redis")


@retry(tries=3)
def main(embed_dim=128, k: int = 100, batch_size: int = 1000):
    volume = Volume(company_id="aboutpet", model="item2vec", version="v1")

    embeddings = load_embeddings(volume, embed_dim=embed_dim)
    items = volume.items(by="pidx")

    unknown_pidxs = [x["pidx"] for x in items.values() if x["name"] == "UNKNOWN"]
    embeddings[unknown_pidxs] = torch.zeros(embed_dim, device=embeddings.device)

    aggregated_scores = collections.defaultdict(list)
    desc = "추천 점수를 계산 및 Redis 할당 중입니다.."
    num_items = embeddings.shape[0]
    for batch_start in tqdm(range(0, num_items, batch_size), desc=desc):
        batch_end = min(batch_start + batch_size, num_items)
        batch_pidxs = list(range(batch_start, batch_end))
        batch_pidxs = [pidx for pidx in batch_pidxs if pidx not in unknown_pidxs]
        if not batch_pidxs:
            continue

        batch_items = [items.get(pidx) for pidx in batch_pidxs]
        batch_categories_1 = [x["category1"] if x else None for x in batch_items]

        similarities = torch.mm(embeddings[batch_pidxs], embeddings.T)

        # Process each item in the batch
        for pidx_in_batch, current_pidx in enumerate(batch_pidxs):
            current_pdid = volume.pidx2pdid(current_pidx)
            current_item = batch_items[pidx_in_batch]
            current_category_1 = batch_categories_1[pidx_in_batch]

            if not current_item:
                continue

            sims = similarities[pidx_in_batch]
            top_k_scores, top_k_pidxs = torch.topk(sims, k * 100)
            top_k_pidxs, top_k_scores = top_k_pidxs.cpu(), top_k_scores.cpu()

            top_k_pidxs, top_k_scores = top_k_pidxs.tolist(), top_k_scores.tolist()
            assert len(top_k_pidxs) == len(top_k_scores)
            for pidx, score in zip(top_k_pidxs, top_k_scores):
                pdid = volume.pidx2pdid(pidx)
                item = items.get(pidx)
                if pidx == current_pidx:
                    continue
                if not item:
                    continue
                if current_category_1 != item["category1"]:
                    continue

                value = {"pdid": pdid, "score": f"{score:.4}"}
                aggregated_scores[current_pdid].append(value)
                if len(aggregated_scores[current_pdid]) >= k:
                    break

            aggregated_scores[current_pdid] = aggregated_scores[current_pdid][:k]

    upload(aggregated_scores)


if __name__ == "__main__":
    main()
