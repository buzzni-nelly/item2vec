from datetime import datetime, timedelta

from item2vec.volume import Migrator


if __name__ == "__main__":
    migrator = Migrator(company_id="aboutpet", model="item2vec", version="v1")
    migrator.migrate_traces(begin_date=datetime(2024, 8, 1))
    migrator.migrate_items()
    migrator.migrate_users()
    migrator.migrate_sequential_pairs()
    migrator.generate_click_purchase_footstep_csv(begin_date=datetime.now() - timedelta(days=7))
    migrator.generate_click_click_footstep_csv(begin_date=datetime.now() - timedelta(days=4))
    migrator.generate_edge_indices_csv()
