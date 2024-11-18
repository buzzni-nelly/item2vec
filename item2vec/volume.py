import collections
import enum
from datetime import datetime, timedelta

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
    start_of_day = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0)
    start_timestamp = start_of_day.timestamp()

    # 날짜의 종료 시간 (23:59:59.999999)
    end_of_day = datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59, 999999)
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
    id = Column(Integer, primary_key=True)
    pid = Column(Integer, index=True)
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
    def dict_items(session: Session):
        items = session.query(Item).all()
        return {item.pdid: item for item in items}

    @staticmethod
    def count(session: Session) -> int:
        return session.query(func.count(Item.id)).scalar()  # count 쿼리 추가

    @staticmethod
    def list_popular_items(session: Session):
        query = """
        SELECT i.pdid, i.purchase_count, i.category1, i.category2
        FROM item i
        JOIN (
            SELECT category1, category2, MAX(purchase_count) AS max_purchase_count
            FROM item
            GROUP BY category1, category2
        ) max_items
        ON i.category1 = max_items.category1
        AND i.category2 = max_items.category2
        AND i.purchase_count = max_items.max_purchase_count
        ORDER BY i.category1, i.purchase_count DESC
        """

        result = session.execute(text(query)).fetchall()
        return result

    def to_dict(self):
        return {
            "id": self.id,
            "pdid": self.pdid,
            "pid": self.pid,
            "purchase_count": self.purchase_count,
            "click_count": self.click_count,
            "name": self.name,
            "category1": self.category1,
            "category2": self.category2,
            "category3": self.category3,
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
        traces = session.query(Trace).all()
        return traces

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
    def get_event_counts_by_pdid(session: Session):
        return (
            session.query(
                Trace.pdid,
                func.sum(
                    case((Trace.event == EventType.purchase.value, 1), else_=0)
                ).label("purchase_count"),
                func.count().label("total_count"),
            )
            .group_by(Trace.pdid)
            .all()
        )


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

        self._items = None
        self._pid2pdid = None
        self._pdid2pid = None

    def aggregate(self):
        return Trace.get_event_counts_by_pdid(self.session)

    def migrate_traces(self, start_date: datetime):
        current_date = start_date

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
        # 집계된 데이터를 가져옵니다
        Item.__table__.drop(self.session.bind, checkfirst=True)
        Item.__table__.create(self.session.bind, checkfirst=True)

        aggregates = self.aggregate()

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

        items, pid = [], 0
        for pdid, purchase_count, click_count in aggregates:
            if pdid in products_dict:
                items.append(
                    Item(
                        pdid=pdid,
                        pid=pid,
                        purchase_count=purchase_count,
                        click_count=click_count,
                        name=products_dict[pdid]["name"],
                        category1=products_dict[pdid]["category1"],
                        category2=products_dict[pdid]["category2"],
                        category3=products_dict[pdid]["category3"],
                    )
                )
                pid += 1

        self.session.bulk_save_objects(items)
        self.session.commit()

    def generate_pairs(
        self, window_size: int = 5, time_delta: int = 60 * 3
    ):
        traces = Trace.list_traces(self.session)
        linked_list = collections.deque(maxlen=window_size)
        item_pairs = []
        for trace in tqdm(traces, desc="Pair data 를 추출 중 입니다.."):
            linked_list.append(trace)

            if len(linked_list) < window_size:
                continue

            pivot = window_size // 2
            current = linked_list[pivot]

            for i in range(len(linked_list)):
                compare = linked_list[i]
                if i == pivot:
                    continue
                if current.user_id != compare.user_id:
                    continue
                if current.timestamp - compare.timestamp > time_delta:
                    continue

                pid_1 = self.pdid2pid(current.pdid)
                pid_2 = self.pdid2pid(compare.pdid)
                if pid_1 and pid_2:
                    item_pairs.append((pid_1, pid_2))
        return item_pairs

    def generate_pairs_csv(self):
        item_pairs = self.generate_pairs()
        pairs_df = pd.DataFrame(item_pairs, columns=["target", "positive"], dtype=int)
        csv_path = self.workspace_path.joinpath(f"item.pairs.csv")
        pairs_df.to_csv(csv_path, index=False)

    def generate_recurrent_edge_indices(self):
        recurrent_edges = []
        items = self.items()
        traces = Trace.list_traces(self.session)
        prev_order, current_order = None, None

        Order = collections.namedtuple("Order", ["user_id", "pid", "category1", "timestamp"])

        for trace in tqdm(traces, desc="recurrent edges 를 추출 중입니다.."):
            current_item = items.get(trace.pdid)
            if not current_item:
                continue

            prev_order, current_order = current_order, Order(
                user_id=trace.user_id,
                pid=current_item["pid"],
                category1=current_item["category1"],
                timestamp=trace.timestamp,
            )

            if not prev_order or not current_order:
                continue
            if prev_order.user_id != current_order.user_id:
                continue
            if prev_order.category1 != current_order.category1:
                continue

            recurrent_edges.append((prev_order.pid, current_order.pid))

        return recurrent_edges


    def generate_purchase_edge_indices(self, linked_list_size: int = 10):
        traces = Trace.list_traces(self.session)
        items = self.items()

        linked_list = collections.deque(maxlen=linked_list_size)
        edge_indices = []

        Order = collections.namedtuple("Order", ["user_id", "pid", "category1", "timestamp"])

        for x in traces:
            current_item = items.get(x.pdid)

            if not current_item:
                continue

            linked_list.append(
                Order(
                    user_id=x.user_id,
                    pid=current_item["pid"],
                    category1=current_item["category1"],
                    timestamp=x.timestamp,
                )
            )

            if x.event not in [EventType.purchase]:
                continue

            current, prev_pids = linked_list[-1], []
            for prev_idx in range(len(linked_list) - 1, -1, -1):
                prev = linked_list[prev_idx]
                if current.user_id != prev.user_id:
                    break
                if current.category1 != prev.category1:
                    break
                if current.timestamp - prev.timestamp > 60 * 10:
                    break
                prev_pids.append(prev.pid)

            for prev_pid in prev_pids:
                if prev_pid != current.pid:
                    edge_indices.append((prev_pid, current.pid))

        print(f"Total edges created: {len(edge_indices)}")
        return edge_indices

    def generate_edge_indices_csv(self):
        edge_indices = self.generate_recurrent_edge_indices()
        edges_df = pd.DataFrame(edge_indices, columns=["source", "target"], dtype=int)
        csv_path = self.workspace_path.joinpath(f"edge.indices.csv")
        edges_df.to_csv(csv_path, index=False)

    def list_popular_items(self):
        return Item.list_popular_items(self.session)

    def items(self) -> dict:
        if not self._items:
            print("Loading items into memory from persistent volume..")
            self._items = {x.pdid: x.to_dict() for x in Item.list_items(self.session)}
        return self._items

    def pid2pdid(self, pid: int) -> str | None:
        if not self._pid2pdid:
            print("Caching pid2pdid..")
            self._pid2pdid = {x["pid"]: x["pdid"] for x in self.items().values()}
        return self._pid2pdid.get(pid)

    def pdid2pid(self, pdid: str) -> int | None:
        if not self._pdid2pid:
            print("Caching pdid2pid..")
            self._pdid2pid = {x["pdid"]: x["pid"] for x in self.items().values()}
        return self._pdid2pid.get(pdid)

    def vocab_size(self) -> int:
        return Item.count(self.session)

    def pdids(self):
        return sorted(self.items().values())

    def pids(self):
        return sorted(item["pid"] for item in self.items().values())


if __name__ == "__main__":
    volume = Volume(site="aboutpet", model="item2vec", version="v1")
    volume.migrate_traces(start_date=datetime(2024, 8, 1))
    volume.migrate_items()
    volume.generate_pairs_csv()
    volume.generate_edge_indices_csv()
