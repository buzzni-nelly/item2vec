ABOUTPET_CLICK_PURCHASE_FOOTSTEP = """
with logs as (
    select
         company_id as mall_id
        , replace(replace(replace(event_info.products[1], '[', ''), ']', ''), '"', '') as pid
        , event_info.products as pids
        , event_type
        , coalesce(case when event_info.user_id = '' then null else event_info.user_id end,brtudid) as uid
        , brtudid
        , url_extract_parameter(event_href, 'recommendationType') as recomm_type
        , url_extract_parameter(event_href, 'recommendationId') as recomm_id
        , time + interval '9' hour as log_time
        , record_id
    from retarget.retarget_access_log ral
    where  "time" >= cast('{date}' as timestamp) - interval '30' minute - interval '9' hour
        -- and "time" <= cast('{date}' as timestamp) - interval '9' hour
        and company_id = 'aboutpet'
        and event_type = 'product'
)
, purs as (
    select
        pur0.*, pi.sale_price
    from (
        select
            coalesce(case when event_info.user_id = '' then null else event_info.user_id end, brtudid) as uid
            , max(time + interval '9' hour) as dt
            , cast(max(time + interval '9' hour) as date) as dt_date
            , replace(replace(replace(pid, '[', ''), ']', ''), '"', '') as pid
            , count(*) as pv
        from retarget.retarget_access_log r
        cross join unnest(event_info.products) as t(pid)
        where "time" >= date_trunc('day', cast(substring('{date}', 1, 10) as timestamp)) - interval '9' hour
            --and "time" < date_trunc('day', cast(substring('{date}', 1, 10) as timestamp)) - interval '9' hour + interval '1'day
            and company_id = 'aboutpet'
            and event_type = 'purchase'
        group by 
            coalesce(case when event_info.user_id = '' then null else event_info.user_id end, brtudid)
            , replace(replace(replace(pid, '[', ''), ']', ''), '"', '')
    ) pur0
    left join analyserdb.public.pdid_aboutpet pi on 'aboutpet_'||pur0.pid = pi.pdid
)
select
    dt_date, uid, brtudid, pid, lead_pid as target_pid, lead_recomm_type as target_recomm_type, log_time as pid_time, lead_time as target_pid_time, view_order_type, purchase_time, purchase_pv, purchase_pv*sale_price as sales, diff_pur
    , pid_preceding, log_time_preceding
    -- dt_date,mall_id,pid,uid,brtudid,recomm_type,recomm_id,log_time,record_id,lead_pid,lead_recomm,lead_time,view_order,view_order_type,view_between_next
    -- ,purchase_time,diff_pur, purchase_pv, purchase_pv*sale_price as sales
    -- ,case when row_purchase = 1 then purchase_pv else null end as purchase_pv_1stview
    -- ,case when row_purchase = 1 then purchase_pv*sale_price  else null end as sales_1stview
from (
    select
        *
        , case when purchase_pv is not null then row_number () over (partition by dt_date, coalesce(uid, brtudid), pid order by log_time) else null end as row_purchase
    from (
        select
            z.*
            , case when view_order = 1 and lead_pid is null then 'only_one'
                when view_order = 1 then 'first'
                when view_order>1 and lead_pid is null then 'last' end view_order_type
            , date_diff('second', z.log_time, z.lead_time)  as view_between_next
            , p.dt as purchase_time, p.pv as purchase_pv
            , date_diff('second', z.lead_time, p.dt) as diff_pur
            , p.sale_price
            , cast(z.log_time as date) as dt_date
        from (
            select
                *
                , lead(pid, 1, null)
                    over (partition by uid order by log_time) as lead_pid
                -- , lag(pid, 1, null)
                --     over (partition by uid order by log_time) as lag_pid
                , lead(case when event_type = 'product' then recomm_type else null end, 1, null)
                    over (partition by uid order by log_time) as lead_recomm_type
                , lead(case when event_type = 'product' then log_time else null end, 1, null)
                    over (partition by uid order by log_time) as lead_time
                , array_agg(pid)
                    over (partition by uid order by case when event_type = 'product' then 1 else 0 end, log_time rows between 20 preceding and current row) as pid_preceding
                , array_agg(log_time)
                    over (partition by uid order by case when event_type = 'product' then 1 else 0 end, log_time rows between 20 preceding and current row) as log_time_preceding
                ,
                row_number () over (partition by coalesce(uid, brtudid) order by log_time) as view_order
            from logs
            where uid is not null or brtudid is not null
        ) z
        left join purs p on z.uid = p.uid and z.lead_pid = p.pid and z.lead_time <= p.dt
            and cast(z.log_time as date) = p.dt_date
    ) y
) w
"""

ABOUTPET_CLICK_CLICK_FOOTSTEP = """
with logs as (
    select
        company_id as mall_id
        , record_id
        , time + interval '9' hour as log_time
        , replace(replace(replace(pid, '[', ''), ']', ''), '"', '') as pid
        , event_type
        , user_id
        , uid
        , brtudid
        , event_device_id
        , lag_event_type, lag_event_referrer, lag_event_href
        , case when lag_event_href like '%aboutpet.co.kr/shop/home%' then 'home'||'_'||abtest
            when lag_event_href like '%aboutpet.co.kr/goods/indexGoodsDetail%' then 'product_'||abtest end as recomm_type
        , recomm_type_ref
    from (
        select 
            element_at(element_at(event_href_params, 'abtest'), 1) as abtest
            , company_id
            , record_id
            , lag(event_type, 1, null) over (partition by 
                coalesce(case when length(event_info.user_id)>4 then event_info.user_id else null end, brtudid, event_device_id) order by time) as lag_event_type
            , lag(event_referrer, 1, null) over (partition by 
                coalesce(case when length(event_info.user_id)>4 then event_info.user_id else null end, brtudid, event_device_id) order by time) as lag_event_referrer
            , lag(event_href, 1, null) over (partition by 
                coalesce(case when length(event_info.user_id)>4 then event_info.user_id else null end, brtudid, event_device_id) order by time) as lag_event_href
            , coalesce(case when length(event_info.user_id)>4 then event_info.user_id else null end, brtudid, event_device_id) as uid
            , event_info.user_id, event_device_id, brtudid
            , event_type, event_info.products
            , event_href, event_referrer
            , user_agent
            , element_at((event_info.products), 1) as pid
            , time
            , case when event_referrer like 'https://aboutpet.co.kr/shop/home%' 
                    and event_referrer like '%BUZZNI%' then 'home_aplusai'
                when event_referrer like 'https://aboutpet.co.kr/shop/home%'
                    and event_referrer like '%BLUX%' then 'home_blux'
                when event_referrer like 'https://aboutpet.co.kr/goods/indexGoodsDetail%' 
                    and event_referrer like '%BUZZNI%' then 'product_aplusai'
                when event_referrer like 'https://aboutpet.co.kr/goods/indexGoodsDetail%'
                    and event_referrer like '%BLUX%' then 'product_blux' end as recomm_type_ref
        from retarget.retarget_access_log ral
        where  "time" >= cast('{date}' as timestamp) - interval '30' minute - interval '9' hour
            and company_id = 'aboutpet'
    ) z
    where event_type = 'product'
        and (abtest is not null or recomm_type_ref is not null)
        and (lag_event_href like '%aboutpet.co.kr/shop/home%'
            or lag_event_href like '%aboutpet.co.kr/goods/indexGoodsDetail%'
        )
        and user_agent not like '%https://naver.me/spd%'
)
, purs as (
    select
        pur0.*, pi.sale_price
    from (
        select
            coalesce(case when event_info.user_id = '' then null else event_info.user_id end, element_at(cookies, 'brtudid')[1]) as uid
            , max(time + interval '9' hour) as dt
            , cast(max(time + interval '9' hour) as date) as dt_date
            , replace(replace(replace(pid, '[', ''), ']', ''), '"', '') as pid
            , count(*) as pv
        from retarget.retarget_access_log r
        cross join unnest(event_info.products) as t(pid)
        where "time" >= date_trunc('day', cast(substring('{date}', 1, 10) as timestamp)) - interval '9' hour
            and company_id = 'aboutpet'
            and event_type = 'purchase'
        group by 
            coalesce(case when event_info.user_id = '' then null else event_info.user_id end, element_at(cookies, 'brtudid')[1])
            , replace(replace(replace(pid, '[', ''), ']', ''), '"', '')
    ) pur0
    left join analyserdb.public.pdid_aboutpet pi on 'aboutpet_'||pur0.pid = pi.pdid
)
select
    dt_date, uid, brtudid, pid, lead_pid as target_pid, lead_recomm_type as target_recomm_type, log_time as pid_time, lead_time as target_pid_time
    , view_order_type, purchase_time, purchase_pv, purchase_pv*sale_price as sales, diff_pur
    , pid_preceding
    , log_time_preceding
    ,case when row_purchase = 1 then purchase_pv else null end as purchase_pv_1stview
from (
    select
        *
        , case when purchase_pv is not null then row_number () over (partition by dt_date, coalesce(uid, brtudid), pid order by log_time) else null end as row_purchase
    from (
        select
            z.*
            , case when view_order = 1 and lead_pid is null then 'only_one'
                when view_order = 1 then 'first'
                when view_order>1 and lead_pid is null then 'last' end view_order_type
            , date_diff('second', z.log_time, z.lead_time)  as view_between_next
            , p.dt as purchase_time, p.pv as purchase_pv
            , date_diff('second', z.lead_time, p.dt) as diff_pur
            , p.sale_price
            , cast(z.log_time as date) as dt_date
        from (
            select
                *
                , lead(pid, 1, null)
                    over (partition by uid order by log_time) as lead_pid
                , lead(case when event_type = 'product' then coalesce(recomm_type_ref, recomm_type) else null end, 1, null)
                    over (partition by uid order by log_time) as lead_recomm_type
                , lead(case when event_type = 'product' then log_time else null end, 1, null)
                    over (partition by uid order by log_time) as lead_time
                , array_agg(pid)
                    over (partition by uid order by case when event_type = 'product' then 1 else 0 end, log_time rows between 20 preceding and current row) as pid_preceding
                , array_agg(log_time)
                    over (partition by uid order by case when event_type = 'product' then 1 else 0 end, log_time rows between 20 preceding and current row) as log_time_preceding
                , row_number () over (partition by coalesce(uid, brtudid) order by log_time) as view_order
            from logs
            where uid is not null or brtudid is not null
        ) z
        left join purs p on z.uid = p.uid and z.lead_pid = p.pid and z.lead_time <= p.dt
            and cast(z.log_time as date) = p.dt_date
    ) y
) w
where lead_recomm_type like 'product%'"""

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
      AND company_id = '{company_id}'
      AND event_type != 'list'
      AND event_type != 'basketview'
)
SELECT
    u.user_id AS user_id,
    ('{company_id}' || '_' || u.product_id) AS pdid,
    u.timestamp AS timestamp,
    u.event AS event
FROM raw_user_products u
WHERE NULLIF(u.user_id, '') IS NOT NULL
  AND NULLIF(u.product_id, '') IS NOT NULL
  AND u.user_id != '00000000-0000-0000-0000-000000000000'
ORDER BY user_id, timestamp
"""
