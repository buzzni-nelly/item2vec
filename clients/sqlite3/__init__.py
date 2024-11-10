import sqlalchemy.orm
from sqlalchemy import Column, Integer, String, Float

from clients.sqlite3.database import engine, session

Base = sqlalchemy.orm.declarative_base()


class Trace(Base):
    __tablename__ = 'trace'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True)
    pdid = Column(String)
    pid = Column(Integer)
    timestamp = Column(Float)

    def insert_users(self, traces: list[dict]):
        session.bulk_insert_mappings(Trace, traces)


Base.metadata.create_all(engine)


