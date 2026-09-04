"""Plottable-variable catalog.

Aggregates the per-category variable lists into one registry with a short
name for each. `ALL` gives every variable; `get(name)` gives one.
"""
from . import ak8, ak4, event, scores, masses, more

_CATALOG = []

for _mod, _cat in [
    (scores, "score"),
    (ak8, "ak8"),
    (ak4, "ak4"),
    (event, "event"),
    (masses, "mass"),
    (more, "more"),
]:
    for _v in _mod.VARIABLES:
        _v["cat"] = _cat
        _CATALOG.append(_v)

ALL = list(_CATALOG)
_BY_NAME = {v["name"]: v for v in ALL}


def get(name):
    if name not in _BY_NAME:
        raise KeyError(
            f"Unknown variable '{name}'. Available: {sorted(_BY_NAME)}"
        )
    return _BY_NAME[name]


#: ROOT branches needed for the catalog variables (unique, sorted)
BRANCHES = sorted({
    v["branch"] for v in ALL if v["branch"] is not None
})


def by_cat():
    cats = {}
    for v in ALL:
        cats.setdefault(v["cat"], []).append(v)
    return cats
