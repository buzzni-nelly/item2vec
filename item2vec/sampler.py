import abc
import ast
import random

import pandas as pd
import torch
from tqdm import tqdm
from transformers import ElectraModel, ElectraTokenizer

from item2vec.volume import Volume


class NegativeSampler(abc.ABC):

    @abc.abstractmethod
    def sample(self, sample: int, k: int = 9) -> list[int]:
        pass


class RandomNegativeSampler(NegativeSampler, abc.ABC):

    def sample(self, sample: int, k: int = 9) -> list[int]:
        pass


class BertNegativeSampler(NegativeSampler, abc.ABC):
    def __init__(self, volume: Volume):
        negative_path = volume.workspace_path.joinpath("negatives.csv")
        df = pd.read_csv(negative_path)
        df['negatives'] = df['negatives'].apply(ast.literal_eval)
        self.negatives = df['negatives'].tolist()

    def sample(self, sample: int, k: int = 9) -> list[int]:
        return self.negatives[sample]

    @classmethod
    @torch.no_grad()
    def generate(cls, volume: Volume, pretrained_model_name_or_path: str = "monologg/koelectra-base-v3-discriminator", k: int = 1000):
        items = volume.items()
        items = sorted(items.values(), key=lambda x: x['pid'])
        names = [x['name'] for x in items]

        def inference(samples: list[str], chunk_size: int = 128) -> torch.Tensor:
            device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            tokenizer = ElectraTokenizer.from_pretrained(pretrained_model_name_or_path)
            electra = ElectraModel.from_pretrained(pretrained_model_name_or_path).to(device)

            embeddings_list = []
            for i in tqdm(range(0, len(samples), chunk_size), desc="Embedding Batches.."):
                chunk = samples[i:i + chunk_size]
                inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(device)
                outputs = electra(**inputs)
                cls_embeddings = outputs.last_hidden_state.to("mps")[:, 0, :]  # (chunk_size, hidden_size)
                embeddings_list.append(cls_embeddings)
            concat_embeddings = torch.cat(embeddings_list, dim=0)  # (total_items, hidden_size)
            return concat_embeddings

        embeddings = inference(names)

        distant_indices_list = []
        for i in tqdm(range(len(embeddings)), desc="Calculating Distant Indices"):
            sample_embedding = embeddings[i].unsqueeze(0)  # Shape: (1, hidden_size)
            other_embeddings = torch.cat((embeddings[:i], embeddings[i + 1:]))  # Exclude the current sample

            # Calculate cosine similarity with all other items
            similarities = torch.cosine_similarity(sample_embedding, other_embeddings, dim=-1)

            # Select top-k indices with the smallest similarity (most distant)
            _, indices = torch.topk(similarities, k, largest=False)

            # Adjust indices to account for exclusion of the current sample
            adjusted_indices = random.sample(indices.tolist(), k=10)
            distant_indices_list.append(adjusted_indices)

        return distant_indices_list
