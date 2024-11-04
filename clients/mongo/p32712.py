from typing import Optional, Dict, List

from pymongo import MongoClient

username = "root"
password = "Buzzni2012!"
host = "idc.buzzni.com"
port = 32712
mongo_uri = f"mongodb://{username}:{password}@{host}"

client = MongoClient(mongo_uri, port)

DEFAULT_PROJECTION = {
    "_id": 1,
    "name": 1,
    "category1": 1,
    "category2": 1,
    "category3": 1,
    "image": 1,
}


def get_product(
    pdid: str, projection: Optional[Dict] = None, company_id: str = "aboutpet"
) -> Optional[Dict]:
    db = client["aiaas-search"]
    collection = db[f"search-product-{company_id}"]

    query = {"_id": pdid}
    projection = projection or DEFAULT_PROJECTION
    item = collection.find_one(query, projection)
    return item


def list_products(
    pdids: List[str], projection: Optional[Dict] = None, company_id: str = "aboutpet"
) -> List[Dict]:
    db = client["aiaas-search"]
    collection = db[f"search-product-{company_id}"]

    projection = projection or DEFAULT_PROJECTION
    items = collection.find({"_id": {"$in": pdids}}, projection)
    return list(items)
