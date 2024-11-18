from datetime import datetime

from item2vec.volume import Volume


if __name__ == "__main__":
    volume = Volume(site="aboutpet", model="item2vec", version="v1")
    volume.migrate_traces(start_date=datetime(2024, 8, 1))
    volume.migrate_items()
    volume.generate_pairs_csv()
    volume.generate_edge_indices_csv()
