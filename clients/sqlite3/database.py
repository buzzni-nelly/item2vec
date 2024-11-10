import sqlalchemy.orm
from sqlalchemy import create_engine

engine = create_engine('sqlite:///example.db')

Session = sqlalchemy.orm.sessionmaker(bind=engine)

session = Session()
