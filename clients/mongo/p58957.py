from pymongo import MongoClient

USERNAME = "search"

PASSWORD = "buzzni2020"

HOST = "172.28.10.15:58957,172.28.10.18:58957"

MONGO_URI = f"mongodb://{USERNAME}:{PASSWORD}@{HOST}/?authSource=admin&replicaSet=replSet1&serverSelectionTimeoutMS=500&connect=false"

DATABASE_NAME = "hsmoa-service"

SEARCH_PRODUCT_COLLECTION_NAME = "search_product"

client = MongoClient(MONGO_URI)


def list_products(pdids: list[str], projection: dict | None = None):
    database = client[DATABASE_NAME]
    collection = database[SEARCH_PRODUCT_COLLECTION_NAME]
    cursor = collection.find(
        {"_id": {"$in": pdids}},
        projection=projection,
    )
    temp = {}
    for document in cursor:
        temp[document["product_id"]] = document

    candidates = [temp.get(x, None) for x in pdids]
    return candidates
