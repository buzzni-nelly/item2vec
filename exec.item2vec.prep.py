import argparse
from datetime import datetime, timedelta, timezone

import dateutil.parser as dateparser

from item2vec.volume import Migrator


def main(company_id: str, version: str, trace_begin_date: datetime, click2purchase_begin_date: datetime):
    model = "item2vec"
    migrator = Migrator(company_id=company_id, model=model, version=version)
    migrator.migrate_traces(begin_date=trace_begin_date)
    migrator.migrate_items()
    migrator.migrate_users()
    migrator.migrate_categories()
    migrator.migrate_skip_grams()
    migrator.migrate_click2purchase_sequences(begin_date=click2purchase_begin_date)
    migrator.migrate_training_user_histories(offset_seconds=1 * 60 * 60, condition="full")
    migrator.migrate_test_user_histories(offset_seconds=48 * 60 * 60)
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
    parser.add_argument(
        "--trace-begin-date",
        type=str,
        required=False,
        help="시작 날짜를 입력하세요 (예: '2025-01-01T00:00:00Z')."
    )
    parser.add_argument(
        "--trace-period-days",
        type=int,
        required=False,
        help="현재 날짜에서 몇 일 전을 시작 날짜로 설정할지 입력하세요."
    )
    parser.add_argument(
        "--click2purchase-days",
        type=int,
        required=False,
        default=7,
        help="현재 날짜에서 몇 일 전을 시작 날짜로 설정할지 입력하세요."
    )

    args = parser.parse_args()

    if args.trace_begin_date:
        trace_begin_date = dateparser.parse(args.trace_begin_date)
        if trace_begin_date.tzinfo is None:
            trace_begin_date = trace_begin_date.replace(tzinfo=timezone.utc)
        else:
            trace_begin_date = trace_begin_date.astimezone(timezone.utc)
    elif args.period_days:
        current_utc_time = datetime.now(timezone.utc)
        trace_begin_date = current_utc_time - timedelta(days=args.period_days)
    else:
        raise ValueError("start-date or period_days must be specified.")

    print(f"Calculated trace start_date: {trace_begin_date}")

    current_utc_time = datetime.now(timezone.utc)
    click2purchase_begin_date = current_utc_time - timedelta(days=args.click2purchase_days)

    main(
        company_id=args.company_id,
        version=args.version,
        trace_begin_date=trace_begin_date,
        click2purchase_begin_date=click2purchase_begin_date,
    )
