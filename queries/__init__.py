QUERY_USER_ITEMS = """
WITH raw_user_products AS (
    SELECT
        event_info.products[1] AS product_id,
        COALESCE(
            CASE WHEN TRIM(event_info.user_id) = '' THEN NULL ELSE event_info.user_id END,
            CASE WHEN TRIM(event_device_id) = '' THEN NULL ELSE event_device_id END,
            COALESCE(
                url_extract_parameter(event_referrer, 'user_email'),
                url_extract_parameter(event_href, 'user_email')
            ),
            url_extract_parameter(event_referrer, 'hsmoa_userid'),
            element_at(cookies, 'brtudid')[1]
        ) AS user_id,
        to_unixtime(time) AS timestamp,
        event_type AS event
    FROM retarget.retarget_access_log
    WHERE DATE(time) = DATE '{date}'
      AND company_id = 'aboutpet'
      AND event_type != 'list'
      AND event_type != 'basketview'
)
SELECT
    u.user_id AS user_id,
    ('aboutpet' || '_' || u.product_id) AS pdid,
    u.timestamp AS timestamp,
    u.event AS event
FROM raw_user_products u
WHERE NULLIF(u.user_id, '') IS NOT NULL
  AND NULLIF(u.product_id, '') IS NOT NULL
  AND u.user_id != '00000000-0000-0000-0000-000000000000'
ORDER BY user_id, timestamp
"""
