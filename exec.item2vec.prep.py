import argparse
from datetime import datetime, timedelta

from item2vec.volume import Migrator


def main(company_id: str, version: str):
    model = "item2vec"
    migrator = Migrator(company_id=company_id, model=model, version=version)
    migrator.migrate_traces(begin_date=datetime(2024, 8, 1))
    migrator.migrate_items()
    migrator.migrate_users()
    migrator.migrate_categories()
    migrator.migrate_skip_grams()
    migrator.migrate_click2purchase_sequences(begin_date=datetime.now() - timedelta(days=7))
    migrator.migrate_training_user_histories(offset_seconds=2 * 60 * 60, condition="full")
    migrator.migrate_test_user_histories(offset_seconds=2 * 60 * 60)
    migrator.generate_edge_indices_csv()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="데이터 마이그레이션 스크립트 실행")
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
