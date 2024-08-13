query_items = """
WITH distinct_products AS (
    SELECT DISTINCT
        event_info.products[1] AS product_id
    FROM retarget.retarget_access_log
    WHERE DATE(time) BETWEEN DATE '{start_date}' AND DATE '{end_date}'
      AND event_info.products[1] IS NOT NULL
      AND company_id = 'gsshop'
)
SELECT
    product_id,
    DENSE_RANK() OVER (ORDER BY product_id) - 1 AS pid
FROM distinct_products
ORDER BY product_id
"""

query_user_items = """
WITH user_products AS (
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
        ) AS uid,
        to_unixtime(time) AS time
    FROM retarget.retarget_access_log
    WHERE DATE(time) BETWEEN DATE '{start_date}' AND DATE '{end_date}'
      AND company_id = 'gsshop'
)
SELECT
    u.uid,
    u.product_id,
    u.time
FROM user_products u
WHERE uid IS NOT NULL
  AND uid != '00000000-0000-0000-0000-000000000000'
  AND product_id IS NOT NULL
ORDER BY u.uid, u.time
"""
