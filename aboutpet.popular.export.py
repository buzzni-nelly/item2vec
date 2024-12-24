import itertools
import json

import clients
from item2vec.volume import Volume

volume = Volume(company_id="aboutpet", model="item2vec", version="v1")

items = volume.list_popular_items(days=30)

dogs, cats = [], []
for item in items:
    if item[2] == "강아지":
        dogs.append((item[0], item[1]))
    if item[2] == "고양이":
        cats.append((item[0], item[1]))

popular_items = list(itertools.chain.from_iterable(zip(dogs, cats)))
popular_items = [{"pdid": x[0], "score": x[1]} for x in popular_items]

print(items)
print(popular_items)
print(len(popular_items))

clients.redis.aiaas_6.set("popularity:aboutpet:purchase:v1", json.dumps(popular_items), ex=60 * 60 * 24 * 365)
