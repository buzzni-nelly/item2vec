from trino.dbapi import connect as trino_connect


def fetch(query) -> tuple:
    conn = trino_connect(
        host="http://trino.buzzni.com",
        port=80,
        user="admin",
        catalog="iceberg",
        schema="iceberg_search",
        timezone="Asia/Seoul",
    )
    cur = conn.cursor()
    cur.execute(query)
    rows, columns = cur.fetchall(), [c[0] for c in cur.description]
    return rows, columns

