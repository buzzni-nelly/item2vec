import json

from configs import settings

_namespace = {}


def load() -> dict:
    if "items" in _namespace:
        return _namespace["items"]
    path = settings.item_path
    with open(path, "r") as f:
        _namespace["items"] = json.load(f)
    return _namespace["items"]


def size() -> int:
    items = load()
    return len(items)


def product_ids() -> list[str]:
    items = load()
    return list(items.keys())


def pids() -> list[int]:
    items = load()
    return list(items.values())
