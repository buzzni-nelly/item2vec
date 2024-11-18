import itertools
import json

import clients
from item2vec.volume import Volume

volume = Volume(site="aboutpet", model="item2vec", version="v1")

items = volume.list_popular_items()

dogs, cats = [], []
for item in items:
    if item[2] == "강아지":
        dogs.append(item[0])
    if item[2] == "고양이":
        cats.append(item[0])

popular_items = list(itertools.chain.from_iterable(zip(dogs, cats)))
popular_items = [{"pdid": x, "score": -1} for x in popular_items]

print(popular_items)
print(len(popular_items))

clients.redis.aiaas_6.set("popularity:aboutpet:purchase:v1", json.dumps(popular_items), ex=60 * 60 * 24 * 365)
