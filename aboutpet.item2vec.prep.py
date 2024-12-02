from datetime import datetime, timedelta

from item2vec.volume import Volume


if __name__ == "__main__":
    volume = Volume(site="aboutpet", model="item2vec", version="v1")
    volume.migrate_traces(begin_date=datetime(2024, 8, 1))
    volume.migrate_items()
    volume.generate_sequential_pairs_csv()
    volume.generate_click_purchase_footstep_csv(begin_date=datetime.now() - timedelta(days=7))
    # volume.generate_click_click_footstep_csv(begin_date=datetime.now() - timedelta(days=4))
    volume.generate_edge_indices_csv()
