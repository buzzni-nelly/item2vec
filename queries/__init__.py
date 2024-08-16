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
        to_unixtime(time) AS time
    FROM retarget.retarget_access_log
    WHERE DATE(time) = DATE '{date}'
      AND company_id = 'gsshop'
)
SELECT
    DISTINCT u.uid AS uid,
    u.product_id AS product_id,
    MIN(u.time) AS time
FROM raw_user_products u
WHERE NULLIF(u.uid, '') IS NOT NULL
  AND NULLIF(u.product_id, '') IS NOT NULL
  AND u.uid != '00000000-0000-0000-0000-000000000000'
GROUP BY u.uid, u.product_id
ORDER BY uid, time
"""

query_joined_items = """
WITH ranked_products AS (
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
        AND company_id = 'gsshop'
      GROUP BY
        event_info.products [1]
      HAVING 
        COUNT(*) >= 20
    )
)
SELECT
  rp.product_id,
  rp.click_count,
  ep.mall_product_name,
  ep.mall_product_category1,
  ep.mall_product_category2,
  ep.mall_product_category3,
  DENSE_RANK() OVER (ORDER BY rp.product_id) - 1 AS pid
FROM
  ranked_products rp
  LEFT JOIN iceberg_search.search_ep ep ON rp.product_id = ep.mall_product_id
WHERE
  ep.mall_id = 'gsshop'
ORDER BY
  pid
"""
