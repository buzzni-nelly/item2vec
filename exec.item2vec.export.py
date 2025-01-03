import argparse
import collections
import json

import torch
from retry import retry
from tqdm import tqdm

import clients
import directories
from item2vec.configs import Settings
from item2vec.modules import GraphBPRItem2Vec
from item2vec.volume import Volume


RELEASE = "i2i"


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


def load_embeddings(volume: Volume, embed_dim: int = 256, num_layers: int = 2):
    checkpoint_path = volume.checkpoints_dirpath / "last.ckpt"
    purchase_edge_index_path = volume.workspace_path.joinpath("edge.purchase.indices.csv")
    vocab_size = volume.vocab_size()
    item2vec_module = GraphBPRItem2Vec.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        vocab_size=vocab_size,
        purchase_edge_index_path=purchase_edge_index_path,
        embed_dim=embed_dim,
    )
    item2vec_module.setup()
    item2vec_module.eval()
    item2vec_module.freeze()

    embeddings = item2vec_module.get_graph_embeddings(num_layers=num_layers)
    return embeddings


def upload(aggregated_scores: dict, company_id: str, model: str, version: str):
    pipeline = clients.redis.aiaas_6.pipeline()
    for pdid, scores in aggregated_scores.items():
        key = f"{RELEASE}:{company_id}:{model}:{version}:{pdid}"
        pipeline.set(key, json.dumps(scores))
        pipeline.expire(key, 30 * 24 * 60 * 60)
    pipeline.execute()
    print(f"Total {len(aggregated_scores)} items updated on redis")


@retry(tries=3)
def main(company_id: str, version: str, embed_dim=128, k: int = 100, batch_size: int = 1000):
    model = "item2vec"
    config_path = directories.config(company_id, model, version)
    settings = Settings.load(config_path)
    volume = Volume(company_id=company_id, model=model, version=version)

    embeddings = load_embeddings(volume, embed_dim=embed_dim, num_layers=settings.num_layers)
    items = volume.items(by="pidx")

    pidx2pdid = {x.pidx: x.pdid for x in volume.list_all_items()}

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

        similarities = torch.mm(embeddings[batch_pidxs], embeddings.T)

        row_indices = torch.arange(len(batch_pidxs), device=similarities.device)
        col_indices = torch.tensor(batch_pidxs, device=similarities.device)
        similarities[row_indices, col_indices] = float("-inf")

        top_k_scores, top_k_indices = torch.topk(similarities, k, dim=1)
        top_k_scores = top_k_scores.cpu()
        top_k_indices = top_k_indices.cpu()

        for i, current_pidx in enumerate(batch_pidxs):
            current_pdid = pidx2pdid[current_pidx]
            neighbors = [
                {
                    "pdid": pidx2pdid[top_k_indices[i][j].item()],
                    "score": f"{top_k_scores[i][j].item():.4}"
                }
                for j in range(k)
            ]
            aggregated_scores[current_pdid] = neighbors

    upload(aggregated_scores, company_id, model, version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommendation 시스템 실행")
    parser.add_argument(
        "--company-id",
        type=str,
        required=True,
        help="처리할 회사 ID를 입력하세요."
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="모델 버전을 입력하세요."
    )
    args = parser.parse_args()

    main(args.company_id, args.version)
