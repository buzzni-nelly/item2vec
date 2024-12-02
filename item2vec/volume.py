import collections
import enum
import time
from datetime import datetime, timedelta
from typing import Literal

import pandas as pd
import sqlalchemy.orm
import sqlalchemy.orm
from sqlalchemy import Column, Integer, String, Float, Enum, func, case, text, Sequence
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from tqdm import tqdm

import clients
import directories
import queries

Base = sqlalchemy.orm.declarative_base()


def get_start_and_end_timestamps(target_date: datetime):
    # 날짜의 시작 시간 (00:00:00)
    start_of_day = datetime(
        target_date.year, target_date.month, target_date.day, 0, 0, 0
    )
    start_timestamp = start_of_day.timestamp()

    # 날짜의 종료 시간 (23:59:59.999999)
    end_of_day = datetime(
        target_date.year, target_date.month, target_date.day, 23, 59, 59, 999999
    )
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

    df = df[
        (df["user_id"] != df["user_id"].shift()) | (df["pdid"] != df["pdid"].shift())
    ]
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
    def list_items(session: Session):
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
    def list_traces(session: Session):
        raw_query = text(
            """
            SELECT 
                user.uidx AS uidx,
                item.pidx AS pidx,
                trace.event AS event,
                trace.timestamp AS timestamp
            FROM trace
            INNER JOIN user ON trace.user_id = user.user_id
            INNER JOIN item ON trace.pdid = item.pdid
        """
        )
        results = session.execute(raw_query).fetchall()
        return [
            {
                "uidx": row.uidx,
                "pidx": row.pidx,
                "event": row.event,
                "timestamp": row.timestamp,
            }
            for row in results
        ]

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
                func.sum(
                    case((Trace.event == EventType.purchase.value, 1), else_=0)
                ).label("purchase_count"),
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
                func.sum(
                    case((Trace.event == EventType.purchase.value, 1), else_=0)
                ).label("purchase_count"),
                func.count().label("click_count"),
            )
            .group_by(Trace.user_id)
            .having(func.count() >= click_count_threshold)
            .all()
        )

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


class Volume:

    def __init__(self, site: str, model: str, version: str):
        self.site = site
        self.model = model
        self.version = version
        self.workspace_path = directories.workspaces.joinpath(site, model, version)
        self.sqlite3_path = self.workspace_path.joinpath(f"{site}-{model}-{version}.db")

        if not self.workspace_path.exists():
            self.workspace_path.mkdir(parents=True)

        self.engine = create_engine(f"sqlite:///{self.sqlite3_path.as_posix()}")
        SessionMaker = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.session = SessionMaker()

        Base.metadata.create_all(self.engine)

        self._items_by_pdid = None
        self._items_by_pidx = None
        self._pidx2pdid = None
        self._pdid2pidx = None

    def migrate_traces(self, begin_date: datetime):
        current_date = begin_date

        while current_date <= datetime.now():
            start_criteria, end_criteria = get_start_and_end_timestamps(current_date)

            print(f"Validating {current_date.strftime('%Y-%m-%d')}")
            if Trace.has_trace_after(self.session, end_criteria):
                current_date += timedelta(days=1)
                continue

            print(f"Downloading {current_date.strftime('%Y-%m-%d')}")
            date_str = current_date.strftime("%Y-%m-%d")
            query = queries.QUERY_USER_ITEMS.format(date=date_str)
            df = fetch_trino_query(query)

            delete_criteria = float(df["timestamp"].min())
            print(f"Deleting traces after timestamp: {delete_criteria}")
            Trace.delete_traces_after(self.session, delete_criteria)

            print(f"Inserting new traces after timestamp: {delete_criteria}")
            df.to_sql(name="trace", con=self.engine, index=False, if_exists="append")
            print(
                f"Data for {date_str} has been successfully saved to the trace table."
            )

            current_date += timedelta(days=1)

    def migrate_items(self):
        Item.reset_table(self.session)

        aggregates = Trace.aggregate_items(self.session)
        pdids, purchase_counts, click_counts = zip(*aggregates)
        products = clients.mongo.p32712.list_products(pdids)
        products_dict = {
            x["_id"]: {
                "name": x.get("name"),
                "category1": x.get("category1", "UNKNOWN")
                .replace("*", "")
                .replace("_", ""),
                "category2": x.get("category2", "UNKNOWN")
                .replace("*", "")
                .replace("_", ""),
                "category3": x.get("category3", "UNKNOWN")
                .replace("*", "")
                .replace("_", ""),
            }
            for x in products
        }

        items, pidx = [], 0
        for pdid, purchase_count, click_count in aggregates:
            if pdid not in products_dict:
                continue
            items.append(
                Item(
                    pidx=pidx,
                    pdid=pdid,
                    purchase_count=purchase_count,
                    click_count=click_count,
                    name=products_dict[pdid]["name"],
                    category1=products_dict[pdid]["category1"],
                    category2=products_dict[pdid]["category2"],
                    category3=products_dict[pdid]["category3"],
                )
            )
            pidx += 1

        self.session.bulk_save_objects(items)
        self.session.commit()

    def migrate_users(self):
        User.reset_table(self.session)

        aggregates = Trace.aggregate_users(self.session)

        users = []
        for uidx, (user_id, purchase_count, click_count) in enumerate(aggregates):
            users.append(
                User(
                    uidx=uidx,
                    user_id=user_id,
                    purchase_count=purchase_count,
                    click_count=click_count,
                )
            )

        self.session.bulk_save_objects(users)
        self.session.commit()

    def generate_sequential_pairs(self, window_size: int = 5, time_delta: int = 60 * 3):
        traces = Trace.list_traces(self.session)
        queue = collections.deque(maxlen=window_size)
        item_pairs = []
        for trace in tqdm(traces, desc="Sequence Pair data 를 추출 중 입니다.."):
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

    def generate_sequential_pairs_csv(self):
        item_pairs = self.generate_sequential_pairs()
        pairs_df = pd.DataFrame(
            item_pairs, columns=["target", "positive", "is_purchased"], dtype=int
        )
        csv_path = self.workspace_path.joinpath(f"item.sequential.pairs.csv")
        pairs_df.to_csv(csv_path, index=False)

    def generate_purchase_pairs(self, begin_date: datetime) -> list:
        begin_date = begin_date.strftime("%Y-%m-%d")
        query = queries.ABOUTPET_ITEM2ITEM.format(date=begin_date)
        rows, columns = clients.trinox.fetch(query)
        df = pd.DataFrame(rows, columns=columns)
        df = df.dropna(subset=["target_pid"])

        df = df[df["purchase_time"].notna()]

        df["source_pdid"] = df["pid"].apply(lambda x: f"aboutpet_{x}")
        df["target_pdid"] = df["target_pid"].apply(lambda x: f"aboutpet_{x}")
        df = df[["source_pdid", "target_pdid"]]
        df = df[df["source_pdid"] != df["target_pdid"]]
        return df.values.tolist()

    def generate_purchase_pairs_csv(self, begin_date: datetime):
        print("Extracting click-and-purchase item pairs data before purchase.")
        item_pairs = self.generate_purchase_pairs(begin_date=begin_date)
        pairs_df = pd.DataFrame(
            item_pairs, columns=["source_pdid", "target_pdid"], dtype=str
        )
        save_path = self.workspace_path.joinpath("validation.csv")
        pairs_df.to_csv(save_path.as_posix(), index=False)

    def generate_purchase_edge_indices(self):
        validation_path = self.workspace_path.joinpath("validation.csv")
        df = pd.read_csv(validation_path.as_posix())

        df["target"] = df["source_pdid"].apply(self.pdid2pidx)
        df["source"] = df["target_pdid"].apply(self.pdid2pidx)
        df = df.dropna()
        df["target"] = df["target"].astype(int)
        df["source"] = df["source"].astype(int)
        df = df[["source", "target"]]
        return df.values.tolist()

    def generate_edge_indices_csv(self):
        purchase_edge_indices = self.generate_purchase_edge_indices()
        purchase_edges_df = pd.DataFrame(
            purchase_edge_indices, columns=["source", "target"], dtype=int
        )
        purchase_csv_path = self.workspace_path.joinpath(f"edge.purchase.indices.csv")
        purchase_edges_df.to_csv(purchase_csv_path, index=False)

    def generate_source_to_targets(self):
        df = pd.read_csv(
            "/home/buzzni/item2vec/workspaces/aboutpet/item2vec/v1/validation.csv"
        )
        df["source"] = df["source_pdid"].apply(self.pdid2pidx)
        df["target"] = df["target_pdid"].apply(self.pdid2pidx)
        df = df.dropna()
        df["source"] = df["source"].astype(int)
        df["target"] = df["target"].astype(int)
        df = df[["source", "target"]]
        source_to_targets = df.groupby("source")["target"].apply(list).to_dict()
        sorted_source_to_targets = {
            source: sorted(set(targets), key=lambda x: targets.count(x), reverse=True)
            for source, targets in source_to_targets.items()
        }
        return sorted_source_to_targets

    def list_popular_items(self, days: int = 30):
        criteria = time.time() - days * 24 * 60 * 7
        return Trace.list_popular_items(self.session, criteria=criteria)

    def items(self, by="pdid") -> dict:
        if "pdid" == by:
            if not self._items_by_pdid:
                print("Loading items into memory from persistent volume..")
                self._items_by_pdid = {
                    x.pdid: x.to_dict() for x in Item.list_items(self.session)
                }
            return self._items_by_pdid
        else:
            if not self._items_by_pidx:
                print("Loading items into memory from persistent volume..")
                self._items_by_pidx = {
                    x.pidx: x.to_dict() for x in Item.list_items(self.session)
                }
            return self._items_by_pidx

    def pidx2pdid(self, pidx: int) -> str | None:
        if not self._pidx2pdid:
            print("Caching idx2pdid..")
            self._pidx2pdid = {x["pidx"]: x["pdid"] for x in self.items().values()}
        return self._pidx2pdid.get(pidx)

    def pdid2pidx(self, pdid: str) -> int | None:
        if not self._pdid2pidx:
            print("Caching pdid2idx..")
            self._pdid2pidx = {x["pdid"]: x["pidx"] for x in self.items().values()}
        return self._pdid2pidx.get(pdid)

    def vocab_size(self) -> int:
        return Item.count(self.session)

    def pdids(self):
        return sorted(self.items().values())

    def pidxs(self):
        return sorted(item["pidx"] for item in self.items().values())


if __name__ == "__main__":
    volume = Volume(site="aboutpet", model="item2vec", version="v1")
    # volume.migrate_traces(start_date=datetime(2024, 8, 1))
    # volume.migrate_items()
    # volume.migrate_users()
    # volume.generate_pairs_csv()
    # volume.generate_edge_indices_csv()
    volume.generate_source_to_targets_csv()
