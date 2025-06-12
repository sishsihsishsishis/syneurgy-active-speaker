import redis
import time
from config import cfg

pool = redis.ConnectionPool(host=cfg.REDIS_HOST, password=cfg.REDIS_PASSWORD, port=cfg.REDIS_PORT, db=0)
r = redis.StrictRedis(connection_pool= pool)

zset_key = 'video-speaker-detector'
value = 'meeting909:test/video/no_trust.mp4:0.6'

score = time.time()
r.zadd(zset_key, {value: score})
