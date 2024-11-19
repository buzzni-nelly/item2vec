import itertools

import torch
from sqlalchemy import create_engine, text
from tqdm import tqdm

from item2vec.models import GraphBPRItem2VecModule
from item2vec.volume import Volume

# Replace 'your_database_url' with your actual database URL
engine = create_engine('sqlite:////Users/nelly/PycharmProjects/item2vec/workspaces/aboutpet/item2vec/v1/aboutpet-item2vec-v1.db')

# Define the modified query
query = text("""
    SELECT user_id, 
           GROUP_CONCAT(pdid, ',') AS pdids, 
           GROUP_CONCAT(event, ',') AS events
    FROM (
        SELECT user_id, pdid, event
        FROM trace
        ORDER BY user_id, timestamp
    ) AS ordered_trace
    GROUP BY user_id
    HAVING COUNT(pdid) >= 2
""")


volume = Volume(site="aboutpet", model="item2vec", version="v1")

vocab_size = volume.vocab_size()
item2vec_module = GraphBPRItem2VecModule.load_from_checkpoint(
    "/tmp/checkpoints/last.ckpt",
    vocab_size=vocab_size,
    edge_index_path=volume.workspace_path.joinpath("edge.sequential.indices.csv"),
    embedding_dim=128
)
item2vec_module.setup()
item2vec_module.eval()
item2vec_module.freeze()

item2vec = item2vec_module.item2vec
chunk_size = 1000
total_range = item2vec.embeddings.weight.shape[0]
chunks = [torch.arange(i, min(i + chunk_size, total_range)) for i in range(0, total_range, chunk_size)]

negatives_list = []
for pids in chunks:
    _, indices = item2vec.get_similar_pids(pids, k=1000, largest=True)
    indices = indices[:, -10:]
    negatives_list.append(indices.tolist())

negatives_list = list(itertools.chain.from_iterable(negatives_list))

with engine.connect() as connection:
    result = connection.execute(query)

    count = 0

    results = []
    for row in tqdm(result):
        user_id, pdids, events = row
        pdids = pdids.split(',')
        pids = [volume.pdid2pid(x) for x in pdids]
        events = events.split(',')
        zipped = list(zip(*[x for x in zip(pids, events) if x[0]]))
        if not zipped:
            continue

        pids, events = zipped
        for i in range(1, len(pids)):
            start_index = max(0, len(pids) - 50)
            end_index = start_index + i
            history_pids, positive_pid = pids[start_index:end_index], pids[end_index]
            history_pids += (-1,) * int(50 - len(history_pids))  # 길이가 50보다 작으면 -1로 패딩
            _, similar_pids = item2vec.get_similar_pids([positive_pid], k=1000, largest=True)
            negative_pids = similar_pids[-10:].tolist()
            encoded_histories = ",".join(map(str, history_pids))
            encoded_positive = str(positive_pid)
            encoded_negatives = ",".join(map(str, negative_pids))
            results.append((encoded_histories, encoded_positive, encoded_negatives))

    print(len(results))
