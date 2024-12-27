from __future__ import annotations

import collections
import enum
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Type

import pandas as pd
import sqlalchemy.orm
import sqlalchemy.orm
from sqlalchemy import Column, Integer, String, Float, Enum, func, case, text
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from tqdm import tqdm

import clients
import directories
import queries

Base = sqlalchemy.orm.declarative_base()


def get_start_and_end_timestamps(target_date: datetime):
    # 날짜의 시작 시간 (00:00:00)
    start_of_day = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=timezone.utc)
    start_timestamp = start_of_day.timestamp()

    # 날짜의 종료 시간 (23:59:59.999999)
    end_of_day = datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59, 999999, tzinfo=timezone.utc)
    end_timestamp = end_of_day.timestamp()

    return start_timestamp, end_timestamp


def fetch_trino_query(query: str):
    rows, columns = clients.trinox.fetch(query=query)

    df = pd.DataFrame(rows, columns=columns)
    df["pdid"].astype(str)
    df["pdid"] = df["pdid"].str.replace(r'\["?|"?\]', "", regex=True)

    df["user_id"].astype(str)
    df.dropna(inplace=True)

    df = df.sort_values(by=["user_id", "timestamp"])
    df = df[["user_id", "pdid", "timestamp", "event"]]

    df = df[(df["user_id"] != df["user_id"].shift()) | (df["pdid"] != df["pdid"].shift())]
    return df


class EventType(enum.Enum):
    product = "product"
    basket = "basket"
    purchase = "purchase"


class Item(Base):
    __tablename__ = "item"
    pidx = Column(Integer, primary_key=True)
    pdid = Column(String, index=True, nullable=False)
    purchase_count = Column(Integer, nullable=False)
    click_count = Column(Integer, nullable=False)
    name = Column(String, nullable=False)
    category1 = Column(String, nullable=False)
    category2 = Column(String, nullable=False)
    category3 = Column(String, nullable=False)

    @staticmethod
    def get_item_by_pdid(session: Session, pdid: str) -> "Item":
        return session.query(Item).filter(Item.pdid == pdid).scalar()

    @staticmethod
    def get_item_by_pidx(session: Session, pidx: int) -> "Item":
        return session.query(Item).filter(Item.pidx == pidx).scalar()

    @staticmethod
    def list_items(session: Session) -> list:
        return session.query(Item).all()

    @staticmethod
    def dict_items(session: Session, by: Literal["pdid", "pidx"] = ""):
        items = session.query(Item).all()
        if by == "pdid":
            return {item.pdid: item for item in items}
        elif by == "pidx":
            return {item.pidx: item for item in items}
        else:
            raise ValueError("by must be one of pdid or idx.")

    @staticmethod
    def count(session: Session) -> int:
        return session.query(func.count(Item.pidx)).scalar()  # count 쿼리 추가

    @staticmethod
    def reset_table(session: Session):
        Item.__table__.drop(session.bind, checkfirst=True)
        Item.__table__.create(session.bind, checkfirst=True)

    def to_dict(self):
        return {
            "pidx": self.pidx,
            "pdid": self.pdid,
            "purchase_count": self.purchase_count,
            "click_count": self.click_count,
            "name": self.name,
            "category1": self.category1,
            "category2": self.category2,
            "category3": self.category3,
        }


class Category1(Base):
    __tablename__ = "category1"
    cidx = Column(Integer, primary_key=True)
    name = Column(String, index=True, nullable=False)

    @staticmethod
    def reset_table(session: Session):
        Category1.__table__.drop(session.bind, checkfirst=True)
        Category1.__table__.create(session.bind, checkfirst=True)

    @staticmethod
    def get_category_by_cidx(session: Session, cidx: int) -> Type[Category1] | None:
        category = session.query(Category1).filter(Category1.cidx == cidx).one_or_none()
        return category

    @staticmethod
    def get_category_by_name(session: Session, name: str) -> Type[Category1] | None:
        category = session.query(Category1).filter(Category1.name == name).one_or_none()
        return category

    @staticmethod
    def count_categories(session: Session) -> int:
        return session.query(func.count(Category1.cidx)).scalar()


class Category2(Base):
    __tablename__ = "category2"
    cidx = Column(Integer, primary_key=True)
    name = Column(String, index=True, nullable=False)

    @staticmethod
    def reset_table(session: Session):
        Category2.__table__.drop(session.bind, checkfirst=True)
        Category2.__table__.create(session.bind, checkfirst=True)

    @staticmethod
    def get_category_by_cidx(session: Session, cidx: int) -> Type[Category2] | None:
        category = session.query(Category2).filter(Category2.cidx == cidx).one_or_none()
        return category

    @staticmethod
    def get_category_by_name(session: Session, name: str) -> Type[Category2] | None:
        category = session.query(Category2).filter(Category2.name == name).one_or_none()
        return category

    @staticmethod
    def count_categories(session: Session) -> int:
        return session.query(func.count(Category2.cidx)).scalar()


class Category3(Base):
    __tablename__ = "category3"
    cidx = Column(Integer, primary_key=True)
    name = Column(String, index=True, nullable=False)

    @staticmethod
    def reset_table(session: Session):
        Category3.__table__.drop(session.bind, checkfirst=True)
        Category3.__table__.create(session.bind, checkfirst=True)

    @staticmethod
    def get_category_by_cidx(session: Session, cidx: int) -> Type[Category3] | None:
        category = session.query(Category3).filter(Category3.cidx == cidx).one_or_none()
        return category

    @staticmethod
    def get_category_by_name(session: Session, name: str) -> Type[Category3] | None:
        category = session.query(Category3).filter(Category3.name == name).one_or_none()
        return category

    @staticmethod
    def count_categories(session: Session) -> int:
        return session.query(func.count(Category3.cidx)).scalar()


class User(Base):
    __tablename__ = "user"
    uidx = Column(Integer, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    purchase_count = Column(Integer, nullable=False, default=0)
    click_count = Column(Integer, nullable=False, default=0)

    @staticmethod
    def list_users(session: Session):
        return session.query(User).all()

    @staticmethod
    def dict_users(session: Session):
        users = session.query(User).all()
        return {x.user_id: x for x in users}

    @staticmethod
    def count(session: Session) -> int:
        return session.query(func.count(User.uidx)).scalar()

    @staticmethod
    def reset_table(session: Session):
        User.__table__.drop(session.bind, checkfirst=True)
        User.__table__.create(session.bind, checkfirst=True)

    def to_dict(self):
        return {
            "uidx": self.uidx,
            "user_id": self.user_id,
            "purchase_count": self.purchase_count,
            "click_count": self.click_count,
        }


class Trace(Base):
    __tablename__ = "trace"
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    pdid = Column(String, nullable=False)
    event = Column(Enum(EventType), nullable=False)
    timestamp = Column(Float, nullable=False)

    @staticmethod
    def count_traces(session: Session) -> int:
        return session.query(func.count(Trace.id)).scalar()

    @staticmethod
    def list_traces(session: Session, chunk_size: int = 10_000_000, begin_timestamp: float | None = None):
        base_query = """
        SELECT 
            user.uidx AS uidx,
            item.pidx AS pidx,
            trace.event AS event,
            trace.timestamp AS timestamp
        FROM trace
        INNER JOIN user ON trace.user_id = user.user_id
        INNER JOIN item ON trace.pdid = item.pdid
        """
        query_args = {}

        if begin_timestamp:
            base_query += " WHERE trace.timestamp > :timestamp"
            query_args["timestamp"] = begin_timestamp

        raw_query = text(base_query)
        result = session.execute(raw_query, query_args).yield_per(chunk_size)
        for row in result:
            yield {
                "uidx": row.uidx,
                "pidx": row.pidx,
                "event": row.event,
                "timestamp": row.timestamp,
            }

    @staticmethod
    def get_last_trace(session: Session) -> "Trace | None":
        return session.query(Trace).order_by(Trace.id.desc()).limit(1).one_or_none()

    @staticmethod
    def get_last_timestamp(session: Session) -> int | None:
        return session.query(func.max(Trace.timestamp)).scalar()

    @staticmethod
    def insert_traces(session: Session, traces: list[dict]):
        session.bulk_insert_mappings(Trace, traces)

    @staticmethod
    def has_trace_after(session: Session, timestamp: float) -> bool:
        max_timestamp = session.query(func.max(Trace.timestamp)).scalar()
        return max_timestamp is not None and max_timestamp >= timestamp

    @staticmethod
    def delete_traces_after(session: Session, timestamp: float) -> None:
        session.query(Trace).filter(Trace.timestamp >= timestamp).delete(
            synchronize_session=False,
        )
        session.commit()

    @staticmethod
    def aggregate_items(session: Session):
        return (
            session.query(
                Trace.pdid,
                func.sum(case((Trace.event == EventType.purchase.value, 1), else_=0)).label("purchase_count"),
                func.count().label("click_count"),
            )
            .group_by(Trace.pdid)
            .all()
        )

    @staticmethod
    def aggregate_users(session: Session, click_count_threshold: int = 5):
        return (
            session.query(
                Trace.user_id,
                func.sum(case((Trace.event == EventType.purchase.value, 1), else_=0)).label("purchase_count"),
                func.count().label("click_count"),
            )
            .group_by(Trace.user_id)
            .having(func.count() >= click_count_threshold)
            .all()
        )

    @staticmethod
    def aggregate_user_histories(
        session: Session,
        condition: Literal["full", "training", "test"] = "full",
        min_purchase_count: int = 1,
        offset_seconds=2 * 60 * 60,
    ):
        last_timestamp = Trace.get_last_timestamp(session)
        threshold = last_timestamp - offset_seconds

        query = f"""
        SELECT
            user_id,
            GROUP_CONCAT(pidx, ',') AS pidxs,
            GROUP_CONCAT(event, ',') AS events,
            GROUP_CONCAT(cidx_1, ',') AS category1s,
            GROUP_CONCAT(cidx_2, ',') AS category2s,
            GROUP_CONCAT(cidx_3, ',') AS category3s,
            GROUP_CONCAT(unix_timestamp, ',') AS unix_timestamps
        FROM (
            SELECT
                t.user_id AS user_id,
                i.pidx AS pidx,
                t.event AS event,
                COALESCE(c1.cidx, 0) AS cidx_1,
                COALESCE(c2.cidx, 0) AS cidx_2,
                COALESCE(c3.cidx, 0) AS cidx_3,
                t.timestamp AS unix_timestamp
            FROM
                trace AS t
            INNER JOIN item AS i ON t.pdid = i.pdid
            INNER JOIN user AS u ON t.user_id = u.user_id
            LEFT JOIN category1 AS c1 ON i.category1 = c1.name
            LEFT JOIN category2 AS c2 ON i.category2 = c2.name
            LEFT JOIN category3 AS c3 ON i.category3 = c3.name
            WHERE u.purchase_count >= {min_purchase_count}
        ) AS subquery
        GROUP BY
            user_id
        """

        if condition == "training":
            query += f"HAVING MAX(unix_timestamp) < {threshold}"
        elif condition == "test":
            query += f"HAVING MAX(unix_timestamp) >= {threshold}"

        query = text(query)
        results = session.execute(query).fetchall()
        return [
            {
                "user_id": row.user_id,
                "pidxs": row.pidxs,
                "events": row.events,
                "category1s": row.category1s,
                "category2s": row.category2s,
                "category3s": row.category3s,
                "unix_timestamps": row.unix_timestamps,
            }
            for row in tqdm(results, desc="Aggregating user histories..")
        ]

    @staticmethod
    def list_popular_items(session: Session, criteria: float):
        query = text(
            """
            SELECT pdid, MAX(cnt) as cnt, category1, category2
            FROM (
                SELECT a.pdid, a.cnt, b.category1, b.category2
                FROM (
                    SELECT pdid, COUNT(pdid) as cnt
                    FROM trace AS t
                    WHERE t.event = 'purchase'
                    AND t.timestamp > :criteria
                    GROUP BY pdid
                ) AS a
                LEFT JOIN item AS b ON a.pdid = b.pdid
            )
            WHERE category1 IS NOT NULL AND category2 IS NOT NULL
            GROUP BY category1, category2
            ORDER BY category1, cnt DESC ;
            """
        )
        result = session.execute(query, {"criteria": criteria}).fetchall()
        return result


class SkipGram(Base):
    __tablename__ = "skip_gram"
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_pidx = Column(Integer)
    target_pidx = Column(Integer)
    is_purchased = Column(Integer, nullable=False)

    @staticmethod
    def get_skip_gram(session: Session, idx: int):
        return session.query(SkipGram).filter(SkipGram.id == idx).one_or_none()

    @staticmethod
    def count_skip_grams(session: Session) -> int:
        return session.query(func.count(SkipGram.id)).scalar()

    @staticmethod
    def reset_table(session: Session):
        SkipGram.__table__.drop(session.bind, checkfirst=True)
        SkipGram.__table__.create(session.bind, checkfirst=True)


class Click2PurchaseSequence(Base):
    __tablename__ = "click2purchase_sequence"
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_pidx = Column(Integer)
    target_pidx = Column(Integer)

    @staticmethod
    def list_click2purchase_sequences(session: Session) -> list:
        return session.query(Click2PurchaseSequence).all()

    @staticmethod
    def reset_table(session: Session):
        Click2PurchaseSequence.__table__.drop(session.bind, checkfirst=True)
        Click2PurchaseSequence.__table__.create(session.bind, checkfirst=True)


class TrainingUserHistory(Base):
    __tablename__ = "training_user_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pidxs = Column(String)
    category1s = Column(String)
    category2s = Column(String)
    category3s = Column(String)

    @staticmethod
    def get_by_id(session, record_id):
        """Retrieve a record by its ID."""
        return session.query(TrainingUserHistory).filter_by(id=record_id).first()

    @staticmethod
    def count(session):
        """Count the total number of records."""
        return session.query(TrainingUserHistory).count()

    @staticmethod
    def reset_table(session: Session):
        TrainingUserHistory.__table__.drop(session.bind, checkfirst=True)
        TrainingUserHistory.__table__.create(session.bind, checkfirst=True)


class TestUserHistory(Base):
    __tablename__ = "test_user_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pidxs = Column(String)
    category1s = Column(String)
    category2s = Column(String)
    category3s = Column(String)

    @staticmethod
    def get_by_id(session, record_id):
        """Retrieve a record by its ID."""
        return session.query(TestUserHistory).filter_by(id=record_id).first()

    @staticmethod
    def count(session):
        """Count the total number of records."""
        return session.query(TestUserHistory).count()

    @staticmethod
    def reset_table(session: Session):
        TestUserHistory.__table__.drop(session.bind, checkfirst=True)
        TestUserHistory.__table__.create(session.bind, checkfirst=True)


class Migrator:
    def __init__(self, company_id: str, model: str, version: str, workspaces_path: Path = None):
        self.company_id = company_id
        self.model = model
        self.version = version

        workspaces_path = workspaces_path or directories.workspaces
        self.workspace_path = workspaces_path.joinpath(company_id, model, version)
        self.sqlite3_path = self.workspace_path.joinpath(f"{company_id}-{model}.sqlite3")

        if not self.workspace_path.exists():
            self.workspace_path.mkdir(parents=True)

        self.engine = create_engine(f"sqlite:///{self.sqlite3_path.as_posix()}")
        SessionMaker = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.session = SessionMaker()

        Base.metadata.create_all(self.engine)

    def migrate_traces(self, begin_date: datetime):
        current_date = begin_date

        while current_date <= datetime.now(tz=timezone.utc).replace():
            start_criteria, end_criteria = get_start_and_end_timestamps(current_date)

            print(f"Validating {current_date.strftime('%Y-%m-%d')}")
            if Trace.has_trace_after(self.session, end_criteria):
                current_date += timedelta(days=1)
                continue

            print(f"Downloading {current_date.strftime('%Y-%m-%d')}")
            date_str = current_date.strftime("%Y-%m-%d")
            query = queries.QUERY_USER_ITEMS.format(company_id=self.company_id, date=date_str)
            df = fetch_trino_query(query)

            delete_criteria = float(df["timestamp"].min())
            print(f"Deleting traces after timestamp: {delete_criteria}")
            Trace.delete_traces_after(self.session, delete_criteria)

            print(f"Inserting new traces after timestamp: {delete_criteria}")
            df.to_sql(name="trace", con=self.engine, index=False, if_exists="append")
            print(f"Data for {date_str} has been successfully saved to the trace table.")

            current_date += timedelta(days=1)

    def migrate_items(self, chunk_size=1000):
        Item.reset_table(self.session)

        aggregates = Trace.aggregate_items(self.session)
        pdids, purchase_counts, click_counts = zip(*aggregates)

        products_dict = {}
        for i in tqdm(range(0, len(pdids), chunk_size), desc="MongoDB 에서 product 정보를 가져옵니다."):
            chunk_pdids = pdids[i : i + chunk_size]
            products = clients.mongo.p32712.list_products(chunk_pdids, company_id=self.company_id)
            products_dict.update(
                {
                    x["_id"]: {
                        "name": x.get("name") or "UNKNOWN",
                        "category1": (x.get("category1") or "UNKNOWN").replace("*", "").replace("_", ""),
                        "category2": (x.get("category2") or "UNKNOWN").replace("*", "").replace("_", ""),
                        "category3": (x.get("category3") or "UNKNOWN").replace("*", "").replace("_", ""),
                    }
                    for x in products
                }
            )

        items = [
            Item(
                pidx=0,
                pdid="UNKNOWN",
                purchase_count=0,
                click_count=0,
                name="UNKNOWN",
                category1="UNKNOWN",
                category2="UNKNOWN",
                category3="UNKNOWN",
            )
        ]
        for pdid, purchase_count, click_count in aggregates:
            if pdid not in products_dict:
                continue
            items.append(
                Item(
                    pdid=pdid,
                    purchase_count=purchase_count,
                    click_count=click_count,
                    name=products_dict[pdid]["name"],
                    category1=products_dict[pdid]["category1"],
                    category2=products_dict[pdid]["category2"],
                    category3=products_dict[pdid]["category3"],
                )
            )

        self.session.bulk_save_objects(items)
        self.session.commit()

    def migrate_users(self):
        User.reset_table(self.session)

        aggregates = Trace.aggregate_users(self.session)

        users = [
            User(
                uidx=0,
                user_id="UNKNOWN",
                purchase_count=0,
                click_count=0,
            )
        ]
        for user_id, purchase_count, click_count in aggregates:
            users.append(
                User(
                    user_id=user_id,
                    purchase_count=purchase_count,
                    click_count=click_count,
                )
            )

        self.session.bulk_save_objects(users)
        self.session.commit()

    def migrate_categories(self):
        # 테이블 초기화
        Category1.reset_table(self.session)
        Category2.reset_table(self.session)
        Category3.reset_table(self.session)

        # UNKNOWN 카테고리 추가
        self.session.add(Category1(cidx=0, name="UNKNOWN"))
        self.session.add(Category2(cidx=0, name="UNKNOWN"))
        self.session.add(Category3(cidx=0, name="UNKNOWN"))
        self.session.commit()

        # distinct category 가져오기
        distinct_category_1 = [x[0] for x in self.session.query(Item.category1).distinct().all()]
        distinct_category_2 = [x[0] for x in self.session.query(Item.category2).distinct().all()]
        distinct_category_3 = [x[0] for x in self.session.query(Item.category3).distinct().all()]

        # UNKNOWN 제외한 실제 카테고리들 정렬
        distinct_category_1 = sorted([c for c in distinct_category_1 if c != "UNKNOWN"])
        distinct_category_2 = sorted([c for c in distinct_category_2 if c != "UNKNOWN"])
        distinct_category_3 = sorted([c for c in distinct_category_3 if c != "UNKNOWN"])

        # 카테고리 삽입
        category1_objects = [Category1(name=c) for c in distinct_category_1]
        category2_objects = [Category2(name=c) for c in distinct_category_2]
        category3_objects = [Category3(name=c) for c in distinct_category_3]

        self.session.bulk_save_objects(category1_objects)
        self.session.bulk_save_objects(category2_objects)
        self.session.bulk_save_objects(category3_objects)
        self.session.commit()

    def migrate_skip_grams(self, window_size: int = 5, chunk_size=10_000_000):
        SkipGram.reset_table(self.session)
        skip_grams = self.list_skip_grams(window_size=window_size)

        rows = []
        insert_query = text(
            """
            INSERT INTO skip_gram (source_pidx, target_pidx, is_purchased)
            VALUES (:source_pidx, :target_pidx, :is_purchased)
            """
        )

        for x, y, z in tqdm(skip_grams, desc="Inserting skip grams into sqlite3.."):
            rows.append({"source_pidx": x, "target_pidx": y, "is_purchased": z})

            if len(rows) >= chunk_size:
                self.session.execute(insert_query, rows)
                self.session.commit()
                rows.clear()

        if rows:
            self.session.execute(insert_query, rows)
            self.session.commit()

        print(f"Migration completed: {len(skip_grams)} rows inserted.")

    def migrate_click2purchase_sequences(self, begin_date: datetime, maxlen: int = 50):
        print("Extracting click2purchase item sequences.")
        Click2PurchaseSequence.reset_table(self.session)
        traces = Trace.list_traces(self.session, begin_timestamp=begin_date.timestamp())
        queue = collections.deque(maxlen=maxlen)
        rows = []
        for current in traces:
            for prev in queue:
                if current["event"] != "purchase":
                    continue
                if current["uidx"] != prev["uidx"]:
                    continue
                if current["timestamp"] > prev["timestamp"] + 600:
                    continue
                rows.append({"source_pidx": prev["pidx"], "target_pidx": current["pidx"]})
            queue.append(current)

        insert_query = text(
            """
            INSERT INTO click2purchase_sequence (source_pidx, target_pidx)
            VALUES (:source_pidx, :target_pidx)
            """
        )

        self.session.execute(insert_query, rows)
        self.session.commit()

        print(f"Migration completed: {len(rows):,} rows inserted.")

    def migrate_training_user_histories(
        self,
        offset_seconds: int = 1 * 24 * 60 * 60,
        chunk_size: int = 100_000,
        max_size: int = 50,
        condition: Literal["full", "training"] = "training",
    ):
        TrainingUserHistory.reset_table(self.session)

        histories = Trace.aggregate_user_histories(
            self.session,
            condition=condition,
            min_purchase_count=1,
            offset_seconds=offset_seconds,
        )

        pidxs_list = [list(map(int, x["pidxs"].split(","))) for x in histories]
        category1s_list = [list(map(int, x["category1s"].split(","))) for x in histories]
        category2s_list = [list(map(int, x["category2s"].split(","))) for x in histories]
        category3s_list = [list(map(int, x["category3s"].split(","))) for x in histories]
        events_list = [list(map(str, x["events"].split(","))) for x in histories]
        timestamps_list = [list(map(float, x["unix_timestamps"].split(","))) for x in histories]

        assert len(pidxs_list) == len(category1s_list) == len(category2s_list) == len(category3s_list) == len(timestamps_list) == len(events_list)

        user_histories = []
        zipped = zip(pidxs_list, category1s_list, category2s_list, category3s_list, events_list, timestamps_list)
        iteration = tqdm(zipped, total=len(pidxs_list), desc=f"Partitioning user histories into 50-sequence windows")
        for pidxs, category1s, category2s, category3s, events, timestamps in iteration:

            assert len(pidxs) == len(category1s) == len(category2s) == len(category3s) == len(events) == len(timestamps)

            for i in range(2, len(pidxs)):

                assert timestamps[-2] < timestamps[-1]

                s = max(0, i - max_size)
                s_pidxs = pidxs[s:i]
                s_category1s = category1s[s:i]
                s_category2s = category2s[s:i]
                s_category3s = category3s[s:i]
                s_events = events[s:i]

                user_histories.append((s_pidxs, s_category1s, s_category2s, s_category3s))

                if s_events[-1] == "purchase":
                    pidx = s_pidxs[-1]
                    category1 = s_category1s[-1]
                    category2 = s_category2s[-1]
                    category3 = s_category3s[-1]
                    for j in range(2, len(s_pidxs)):
                        g_pidxs = s_pidxs[-max_size:j] + [pidx]
                        g_category1 = s_category1s[-max_size:j] + [category1]
                        g_category2 = s_category2s[-max_size:j] + [category2]
                        g_category3 = s_category3s[-max_size:j] + [category3]
                        user_histories.append((g_pidxs, g_category1, g_category2, g_category3))

        rows = []
        insert_query = text(
            """
            INSERT INTO training_user_history (pidxs, category1s, category2s, category3s)
            VALUES (:pidxs, :category1s, :category2s, :category3s)
            """
        )

        progress_bar = tqdm(user_histories, desc="Inserting user histories into sqlite3..")
        for s_pidxs, s_category1s, s_category2s, s_category3s in progress_bar:
            rows.append(
                {
                    "pidxs": ",".join(map(str, s_pidxs)),
                    "category1s": ",".join(map(str, s_category1s)),
                    "category2s": ",".join(map(str, s_category2s)),
                    "category3s": ",".join(map(str, s_category3s)),
                }
            )

            if len(rows) >= chunk_size:
                self.session.execute(insert_query, rows)
                self.session.commit()
                rows.clear()

        if rows:
            self.session.execute(insert_query, rows)
            self.session.commit()

        print(f"Migrating train data completed: {len(user_histories)} rows inserted.")

    def migrate_test_user_histories(self, offset_seconds: int = 1 * 24 * 60 * 60, chunk_size=100_000, max_size: int = 50):
        TestUserHistory.reset_table(self.session)

        histories = Trace.aggregate_user_histories(
            self.session, condition="test", min_purchase_count=0, offset_seconds=offset_seconds
        )

        pidxs_list = [list(map(int, x["pidxs"].split(","))) for x in histories]
        category1s_list = [list(map(int, x["category1s"].split(","))) for x in histories]
        category2s_list = [list(map(int, x["category2s"].split(","))) for x in histories]
        category3s_list = [list(map(int, x["category3s"].split(","))) for x in histories]
        events_list = [list(map(str, x["events"].split(","))) for x in histories]
        timestamps_list = [list(map(float, x["unix_timestamps"].split(","))) for x in histories]

        assert len(pidxs_list) == len(category1s_list) == len(category2s_list) == len(category3s_list) == len(timestamps_list) == len(events_list)

        user_histories = []
        zipped = zip(pidxs_list, category1s_list, category2s_list, category3s_list, events_list, timestamps_list)
        iteration = tqdm(zipped, total=len(pidxs_list), desc=f"Partitioning user histories into 50-sequence windows")
        for pidxs, category1s, category2s, category3s, events, timestamps in iteration:
            user_histories.append((pidxs[-50:], category1s[-50:], category2s[-50:], category3s[-50:]))

        rows = []
        insert_query = text(
            """
            INSERT INTO test_user_history (pidxs, category1s, category2s, category3s)
            VALUES (:pidxs, :category1s, :category2s, :category3s)
            """
        )

        progress_bar = tqdm(user_histories, desc="Inserting user histories into sqlite3..")
        for s_pidxs, s_category1s, s_category2s, s_category3s in progress_bar:
            rows.append(
                {
                    "pidxs": ",".join(map(str, s_pidxs)),
                    "category1s": ",".join(map(str, s_category1s)),
                    "category2s": ",".join(map(str, s_category2s)),
                    "category3s": ",".join(map(str, s_category3s)),
                }
            )

            if len(rows) >= chunk_size:
                self.session.execute(insert_query, rows)
                self.session.commit()
                rows.clear()

        if rows:
            self.session.execute(insert_query, rows)
            self.session.commit()

        print(f"Migrating test data completed: {len(user_histories)} rows inserted.")

    def list_user_histories(
        self,
        condition: Literal["full", "training", "test"] = "training",
        min_purchase_count: int = 1,
        offset_seconds: int = 2 * 60 * 60,
        max_size: int = 50,
    ):
        histories = Trace.aggregate_user_histories(
            self.session, condition=condition, min_purchase_count=min_purchase_count, offset_seconds=offset_seconds
        )
        pidxs_list = [list(map(int, x["pidxs"].split(","))) for x in histories]
        category1s_list = [list(map(int, x["category1s"].split(","))) for x in histories]
        category2s_list = [list(map(int, x["category2s"].split(","))) for x in histories]
        category3s_list = [list(map(int, x["category3s"].split(","))) for x in histories]
        events_list = [list(map(str, x["events"].split(","))) for x in histories]
        timestamps_list = [list(map(float, x["unix_timestamps"].split(","))) for x in histories]

        assert len(pidxs_list) == len(category1s_list) == len(category2s_list) == len(category3s_list) == len(timestamps_list) == len(events_list)

        results = []
        zipped = zip(pidxs_list, category1s_list, category2s_list, category3s_list, events_list, timestamps_list)
        iteration = tqdm(zipped, total=len(pidxs_list), desc=f"Partitioning user histories into 50-sequence windows")
        for pidxs, category1s, category2s, category3s, events, timestamps in iteration:

            assert len(pidxs) == len(category1s) == len(category2s) == len(category3s) == len(events) == len(timestamps)

            for i in range(2, len(pidxs)):

                assert timestamps[-2] < timestamps[-1]

                s = max(0, i - max_size)
                s_pidxs = pidxs[s:i]
                s_category1s = category1s[s:i]
                s_category2s = category2s[s:i]
                s_category3s = category3s[s:i]
                s_events = events[s:i]

                if condition != "test":
                    results.append((s_pidxs, s_category1s, s_category2s, s_category3s))

                if s_events[-1] == "purchase":
                    pidx = s_pidxs[-1]
                    category1 = s_category1s[-1]
                    category2 = s_category2s[-1]
                    category3 = s_category3s[-1]
                    for j in range(2, len(s_pidxs)):
                        g_pidxs = s_pidxs[-max_size:j] + [pidx]
                        g_category1 = s_category1s[-max_size:j] + [category1]
                        g_category2 = s_category2s[-max_size:j] + [category2]
                        g_category3 = s_category3s[-max_size:j] + [category3]
                        results.append((g_pidxs, g_category1, g_category2, g_category3))

        print(f">>> List user histories completed.")
        print(f">>> Aggregated total {len(results):,} rows")
        return results

    def list_skip_grams(self, window_size: int = 5, time_delta: int = 60 * 3):
        traces = Trace.list_traces(self.session)
        queue = collections.deque(maxlen=window_size)
        item_pairs = []
        for trace in tqdm(traces, desc="Skip Gram data 를 추출 중 입니다.."):
            queue.append(trace)

            if len(queue) < window_size:
                continue

            pivot = window_size // 2
            current = queue[pivot]

            for i in range(len(queue)):
                compare = queue[i]
                if i == pivot:
                    continue
                if current["uidx"] != compare["uidx"]:
                    continue
                if current["timestamp"] - compare["timestamp"] > time_delta:
                    continue

                pidx_1 = current["pidx"]
                pidx_2 = compare["pidx"]
                weight = 1 if "purchase" in [current["event"], compare["event"]] else 0

                if pidx_1 and pidx_2:
                    item_pairs.append((pidx_1, pidx_2, weight))
                    item_pairs.append((pidx_2, pidx_1, weight))

        return item_pairs

    def pidx2pdid(self, pidx: int) -> str | None:
        item = Item.get_item_by_pidx(self.session, pidx)
        return item.pdid if item else None

    def pdid2pidx(self, pdid: str) -> int | None:
        item = Item.get_item_by_pdid(self.session, pdid)
        return item.pidx if item else None

    def generate_edge_indices_csv(self):
        sequences = Click2PurchaseSequence.list_click2purchase_sequences(self.session)
        # swap source & target
        sequences = [(x.target_pidx, x.source_pidx) for x in sequences]
        purchase_edges_df = pd.DataFrame(sequences, columns=["source_pidx", "target_pidx"], dtype=int)
        purchase_csv_path = self.workspace_path.joinpath(f"edge.purchase.indices.csv")
        purchase_edges_df.to_csv(purchase_csv_path, index=False)


class Volume:

    def __init__(self, company_id: str, model: str, version: str, workspaces_path: Path = None):
        self.company_id = company_id
        self.model = model
        self.version = version

        workspaces_path = workspaces_path or directories.workspaces
        self.workspaces_path = workspaces_path
        self.workspace_path = workspaces_path.joinpath(company_id, model, version)
        self.sqlite3_path = self.workspace_path.joinpath(f"{company_id}-{model}.sqlite3")
        self.onnx_path = self.workspace_path.joinpath(f"{company_id}-{model}.onnx")
        self.checkpoints_dirpath = self.workspace_path.joinpath("checkpoints")

        if not self.workspace_path.exists():
            self.workspace_path.mkdir(parents=True)

        self.engine = create_engine(f"sqlite:///{self.sqlite3_path.as_posix()}")
        SessionMaker = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.session = SessionMaker()

        self._items_by_pdid = None
        self._items_by_pidx = None

    def list_popular_items(self, days: int = 30):
        criteria = time.time() - days * 24 * 60 * 7
        return Trace.list_popular_items(self.session, criteria=criteria)

    def items(self, by="pdid") -> dict:
        if by == "pdid":
            return {x.pdid: x.to_dict() for x in Item.list_items(self.session)}
        else:
            return {x.pidx: x.to_dict() for x in Item.list_items(self.session)}

    def pidx2pdid(self, pidx: int) -> str | None:
        item = Item.get_item_by_pidx(self.session, pidx)
        return item.pdid if item else None

    def pdid2pidx(self, pdid: str) -> int | None:
        item = Item.get_item_by_pdid(self.session, pdid)
        return item.pidx if item else None

    def vocab_size(self) -> int:
        return Item.count(self.session)

    def pdids(self) -> list[str]:
        items = Item.list_items(self.session)
        return [x.pdid for x in items]

    def pidxs(self) -> list[int]:
        items = Item.list_items(self.session)
        return [x.pidx for x in items]

    def get_skip_gram(self, idx: int):
        return SkipGram.get_skip_gram(self.session, idx)

    def count_skip_grams(self):
        return SkipGram.count_skip_grams(self.session)

    def get_category_1_by_cidx(self, cidx: int) -> Type[Category1] | None:
        category = Category1.get_category_by_cidx(self.session, cidx)
        if not category:
            return Category1.get_category_by_cidx(self.session, 0)
        return category

    def get_category_2_by_cidx(self, cidx: int) -> Type[Category2] | None:
        category = Category2.get_category_by_cidx(self.session, cidx)
        if not category:
            return Category2.get_category_by_cidx(self.session, 0)
        return category

    def get_category_3_by_cidx(self, cidx: int) -> Type[Category3] | None:
        category = Category3.get_category_by_cidx(self.session, cidx)
        if not category:
            return Category3.get_category_by_cidx(self.session, 0)
        return category

    def get_category_1_by_name(self, name: str) -> Type[Category1] | None:
        category = Category1.get_category_by_name(self.session, name)
        if not category:
            return Category1.get_category_by_cidx(self.session, 0)
        return category

    def get_category_2_by_name(self, name: str) -> Type[Category2] | None:
        category = Category2.get_category_by_name(self.session, name)
        if not category:
            return Category2.get_category_by_cidx(self.session, 0)
        return category

    def get_category_3_by_name(self, name: str) -> Type[Category3] | None:
        category = Category3.get_category_by_name(self.session, name)
        if not category:
            return Category3.get_category_by_cidx(self.session, 0)
        return category

    def count_categories(self) -> tuple[int, int, int]:
        return self.count_category_1(), self.count_category_2(), self.count_category_3()

    def count_category_1(self) -> int:
        return Category1.count_categories(self.session)

    def count_category_2(self) -> int:
        return Category2.count_categories(self.session)

    def count_category_3(self) -> int:
        return Category3.count_categories(self.session)

    def list_user_histories(
        self,
        condition: Literal["full", "training", "test"] = "training",
        min_purchase_count: int = 1,
    ):
        histories = Trace.aggregate_user_histories(
            self.session,
            condition=condition,
            min_purchase_count=min_purchase_count,
        )
        pidxs_list = [list(map(int, x["pidxs"].split(","))) for x in histories]
        category1s_list = [list(map(int, x["category1s"].split(","))) for x in histories]
        category2s_list = [list(map(int, x["category2s"].split(","))) for x in histories]
        category3s_list = [list(map(int, x["category3s"].split(","))) for x in histories]

        assert len(pidxs_list) == len(category1s_list) == len(category2s_list) == len(category3s_list)

        result = []
        zipped = zip(pidxs_list, category1s_list, category2s_list, category3s_list)
        iteration = tqdm(zipped, total=len(pidxs_list), desc=f"Partitioning user histories into 50-sequence windows")
        for pidxs, category1s, category2s, category3s in iteration:
            assert len(pidxs) == len(category1s) == len(category2s) == len(category3s)
            for i in range(2, len(pidxs)):
                s = max(0, i - 50)
                s_pidxs = pidxs[s:i]
                s_category1s = category1s[s:i]
                s_category2s = category2s[s:i]
                s_category3s = category3s[s:i]
                result.append((s_pidxs, s_category1s, s_category2s, s_category3s))

        print(f">>> List user histories completed.")
        print(f">>> Aggregated total {len(result):,} rows")
        return result

    def get_user_history_count(self, condition: Literal["training", "test"] = "training") -> int:
        if condition == "training":
            return TrainingUserHistory.count(self.session)
        else:
            return TestUserHistory.count(self.session)

    def get_user_history(self, record_id, condition: Literal["training", "test"] = "training") -> tuple[list[int], ...]:
        if condition == "training":
            hist = TrainingUserHistory.get_by_id(self.session, record_id)
        else:
            hist = TestUserHistory.get_by_id(self.session, record_id)

        pidxs = list(map(int, hist.pidxs.split(",")))
        cat1_cidxs = list(map(int, hist.category1s.split(",")))
        cat2_cidxs = list(map(int, hist.category2s.split(",")))
        cat3_cidxs = list(map(int, hist.category3s.split(",")))
        return pidxs, cat1_cidxs, cat2_cidxs, cat3_cidxs

    def list_click2purchase_sequences(self) -> list[Click2PurchaseSequence]:
        return Click2PurchaseSequence.list_click2purchase_sequences(self.session)


if __name__ == "__main__":
    migrator = Migrator(company_id="aboutpet", model="item2vec", version="v1")
    migrator.migrate_traces(begin_date=datetime(2024, 8, 1, tzinfo=timezone.utc))
    migrator.migrate_items()
    migrator.migrate_users()
    migrator.migrate_categories()
    migrator.migrate_skip_grams()
    migrator.migrate_click2purchase_sequences(begin_date=datetime.now() - timedelta(days=7))
    migrator.migrate_training_user_histories(condition="full", offset_seconds=6 * 60 * 60)
    migrator.migrate_test_user_histories(offset_seconds=1 * 24 * 60 * 60)
    migrator.generate_edge_indices_csv()
