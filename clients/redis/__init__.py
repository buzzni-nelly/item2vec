import redis

HOST = "aiaas-service-001.kynlrz.0001.apn2.cache.amazonaws.com"

aiaas_6 = redis.Redis(
    connection_pool=redis.ConnectionPool(host=HOST, db=6),
    socket_connect_timeout=3,
    socket_timeout=3,
    socket_keepalive=True,
)
