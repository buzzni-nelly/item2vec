from datetime import datetime, timedelta

import directories
from item2vec.configs import Settings as Item2VecSettings
from item2vec.volume import Migrator


if __name__ == "__main__":
    config_path = directories.config(company_id="gsshop", model="item2vec", version="v1")
    settings = Item2VecSettings.load(config_path)

    migrator = Migrator(company_id="gsshop", model="item2vec", version="v1")
    # migrator.migrate_traces(begin_date=datetime.now() - timedelta(days=13))
    # migrator.migrate_items()
    # migrator.migrate_users()
    # migrator.migrate_categories()
    # migrator.migrate_skip_grams(window_size=settings.skipgram_window_size)
    # migrator.migrate_click2purchase_sequences(begin_date=datetime.now() - timedelta(days=2))
    migrator.migrate_training_user_histories(offset_seconds=1 * 60 * 60)
    migrator.migrate_test_user_histories(offset_seconds=1 * 60 * 60)
    migrator.generate_edge_indices_csv()
