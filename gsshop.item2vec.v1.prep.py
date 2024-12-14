from datetime import datetime, timedelta

from item2vec.volume import Migrator


if __name__ == "__main__":
    migrator = Migrator(company_id="gsshop", model="item2vec", version="v1")
    migrator.migrate_traces(begin_date=datetime(2024, 11, 29))
    migrator.migrate_items()
    migrator.migrate_users()
    migrator.migrate_skip_grams()
    migrator.migrate_click2purchase_sequences(begin_date=datetime.now() - timedelta(days=2))
    migrator.generate_edge_indices_csv()
