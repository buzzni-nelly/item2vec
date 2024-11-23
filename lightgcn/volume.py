import enum
from datetime import datetime, timedelta

import enum
from datetime import datetime, timedelta

import pandas as pd
import sqlalchemy.orm
import sqlalchemy.orm
from sqlalchemy import Column, Integer, String, Float, Enum, func, case, Sequence, text
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

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
    id = Column(Integer,Sequence('id_sequence', start=0, increment=1), primary_key=True)
    pdid = Column(String, index=True, nullable=False)
    purchase_count = Column(Integer, nullable=False, default=0)
    click_count = Column(Integer, nullable=False, default=0)
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
    def reset_table(session: Session):
        Item.__table__.drop(session.bind, checkfirst=True)
        Item.__table__.create(session.bind, checkfirst=True)

    def to_dict(self):
        return {
            "id": self.id,
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
    id = Column(Integer,Sequence('id_sequence', start=-1, increment=1), primary_key=True)
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
        return session.query(func.count(User.id)).scalar()

    @staticmethod
    def reset_table(session: Session):
        User.__table__.drop(session.bind, checkfirst=True)
        User.__table__.create(session.bind, checkfirst=True)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "purchase_count": self.purchase_count,
            "click_count": self.click_count,
        }


class Trace(Base):
    __tablename__ = "trace"
    id = Column(Integer, Sequence('id_sequence', start=0, increment=1), primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    pdid = Column(String, nullable=False)
    event = Column(Enum(EventType), nullable=False)
    timestamp = Column(Float, nullable=False)

    @staticmethod
    def list_traces(session: Session):
        # SQL 쿼리 작성
        raw_query = text("""
            SELECT 
                user.id AS user_id,
                item.id AS item_id,
                trace.event AS event,
                trace.timestamp AS timestamp
            FROM trace
            INNER JOIN user ON trace.user_id = user.user_id
            INNER JOIN item ON trace.pdid = item.pdid
        """)

        # 실행 및 데이터 반환
        results = session.execute(raw_query).fetchall()

        # 데이터 변환
        return [
            {
                "user_id": row.user_id,
                "item_id": row.item_id,
                "event": row.event,
                "timestamp": row.timestamp,
            }
            for row in results
        ]

    @staticmethod
    def count(session: Session) -> int:
        return session.query(func.count(Trace.id)).scalar()

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
                func.count().label("total_count"),
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
                func.count().label("total_count"),
            )
            .group_by(Trace.user_id)
            .having(func.count() >= click_count_threshold)
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

    def list_traces(self):
        return Trace.list_traces(self.session)

    def count_items(self):
        return Item.count(self.session)

    def count_users(self):
        return User.count(self.session)

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

        items, idx = [], 0
        for pdid, purchase_count, click_count in aggregates:
            if pdid not in products_dict:
                continue
            items.append(
                Item(
                    id=idx,
                    pdid=pdid,
                    purchase_count=purchase_count,
                    click_count=click_count,
                    name=products_dict[pdid]["name"],
                    category1=products_dict[pdid]["category1"],
                    category2=products_dict[pdid]["category2"],
                    category3=products_dict[pdid]["category3"],
                )
            )
            idx += 1

        self.session.bulk_save_objects(items)
        self.session.commit()

    def migrate_users(self):
        User.reset_table(self.session)

        aggregates = Trace.aggregate_users(self.session)

        users = []
        for idx, (user_id, purchase_count, click_count) in enumerate(aggregates):
            users.append(
                User(
                    id=idx,
                    user_id=user_id,
                    purchase_count=purchase_count,
                    click_count=click_count,
                )
            )

        self.session.bulk_save_objects(users)
        self.session.commit()

    def generate_click_edge_indices(self) -> list[tuple[int, int]]:
        user_count = self.count_users()
        traces = self.list_traces()

        edges = []
        for trace in traces:
            user_id = trace["user_id"]
            item_id = trace["item_id"] + user_count
            edges.append((user_id, item_id))
            edges.append((item_id, user_id))

        return list(set(edges))

    def generate_purchase_edge_indices(self) -> list[tuple[int, int]]:
        user_count = self.count_users()
        traces = self.list_traces()

        edges = []
        for trace in traces:
            if trace["event"] != "purchase":
                continue
            user_id = trace["user_id"]
            item_id = trace["item_id"] + user_count
            edges.append((user_id, item_id))

        return list(set(edges))

    def generate_edge_indices_csv(self):
        click_edge_indices = self.generate_click_edge_indices()
        click_edges_df = pd.DataFrame(click_edge_indices, columns=["source", "target"], dtype=int)
        click_csv_path = self.workspace_path.joinpath(f"edge.click.indices.csv")
        click_edges_df.to_csv(click_csv_path, index=False)

        purchase_edge_indices = self.generate_purchase_edge_indices()
        purchase_edges_df = pd.DataFrame(purchase_edge_indices, columns=["source", "target"], dtype=int)
        purchase_csv_path = self.workspace_path.joinpath(f"edge.purchase.indices.csv")
        purchase_edges_df.to_csv(purchase_csv_path, index=False)


if __name__ == "__main__":
    volume = Volume(site="aboutpet", model="lightgcn", version="v1")
    volume.migrate_traces(start_date=datetime(2024, 8, 1))
    volume.migrate_items()
    volume.migrate_users()
    volume.generate_edge_indices_csv()
