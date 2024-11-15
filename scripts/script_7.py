from sqlalchemy import create_engine, text

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
    edge_index_path=volume.workspace_path.joinpath("edge.indices.csv"),
    embedding_dim=128
)
item2vec_module.setup()
item2vec_module.eval()
item2vec_module.freeze()

item2vec = item2vec_module.item2vec

# Execute the query
with engine.connect() as connection:
    result = connection.execute(query)

    count = 0

    for row in result:
        user_id, pdids, events = row
        pdids = pdids.split(',')
        pids = [volume.pdid2pid(x) for x in pdids]
        events = events.split(',')
        zipped = list(zip(*[x for x in zip(pids, events) if x[0]]))
        if not zipped:
            continue

        pids, events = zipped
        for i in range(1, len(pids)):
            history_pids, target_pid = pids[:i], pids[i]
            _, similar_pids = item2vec.get_similar_pids(target_pid, k=10000, largest=True)
            close_negatives = similar_pids[101:105].tolist()
            distant_negatives = similar_pids[9995: 10000].tolist()
            negatives = close_negatives + distant_negatives
            print(similar_pids)
    print(count)
