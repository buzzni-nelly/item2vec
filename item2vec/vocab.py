import json

import directories

_namespace = {}


def load() -> dict:
    if "items" in _namespace:
        return _namespace["items"]
    with open(directories.item, "r") as f:
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
