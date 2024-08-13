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


"""
    SELECT
        pid,
        COALESCE(
            CASE WHEN TRIM(event_info.user_id) = '' THEN NULL ELSE event_info.user_id END,
            CASE WHEN TRIM(event_device_id) = '' THEN NULL ELSE event_device_id END,
            COALESCE(
                url_extract_parameter(event_referrer, 'user_email'),
                url_extract_parameter(event_href, 'user_email')
            ),
            url_extract_parameter(event_referrer, 'hsmoa_userid'),
            element_at(cookies, 'brtudid')[1]
        ) AS uid,
        time,
    FROM retarget.retarget_access_log
"""
