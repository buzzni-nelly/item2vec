from datetime import datetime, timedelta

from item2vec.volume import Migrator


if __name__ == "__main__":
    migrator = Migrator(company_id="gsshop", model="item2vec", version="v1")
    migrator.migrate_traces(begin_date=datetime.now() - timedelta(days=13))
    migrator.migrate_items()
    migrator.migrate_users()
    migrator.migrate_categories()
    migrator.migrate_skip_grams(window_size=3)
    migrator.migrate_click2purchase_sequences(begin_date=datetime.now() - timedelta(days=3))
    migrator.migrate_training_user_histories()
    migrator.migrate_test_user_histories()
    migrator.generate_edge_indices_csv()
