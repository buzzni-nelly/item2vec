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
        ) AS uid,
        to_unixtime(time) AS time,
        event_type AS event
    FROM retarget.retarget_access_log
    WHERE DATE(time) = DATE '{date}'
      AND company_id = 'aboutpet'
      AND event_type != 'list'
      AND event_type != 'basketview'
)
SELECT
    u.uid AS uid,
    ('aboutpet' || '_' || u.product_id) AS pdid,
    u.time AS time,
    u.event AS event
FROM raw_user_products u
WHERE NULLIF(u.uid, '') IS NOT NULL
  AND NULLIF(u.product_id, '') IS NOT NULL
  AND u.uid != '00000000-0000-0000-0000-000000000000'
ORDER BY uid, time
"""

QUERY_ITEMS = """
WITH product AS (
  SELECT
    product_id,
    click_count
  FROM
    (
      SELECT
        event_info.products [1] AS product_id,
        COUNT(event_info.products [1]) AS click_count
      FROM
        retarget.retarget_access_log
      WHERE
        DATE(time) BETWEEN DATE '{start_date}' AND DATE '{end_date}'
        AND event_info.products [1] IS NOT NULL
        AND event_info.products [1] != ''
        AND company_id = 'aboutpet'
      GROUP BY
        event_info.products [1]
      HAVING 
        COUNT(*) >= 100
    )
)
SELECT
  p.product_id,
  p.click_count,
  ep.mall_product_name,
  ep.mall_product_category1,
  ep.mall_product_category2,
  ep.mall_product_category3,
  DENSE_RANK() OVER (ORDER BY p.product_id) - 1 AS pid
FROM
  product p
  LEFT JOIN iceberg_search.search_ep ep ON p.product_id = ep.mall_product_id
WHERE
  ep.mall_id = 'aboutpet'
ORDER BY
  pid
"""
