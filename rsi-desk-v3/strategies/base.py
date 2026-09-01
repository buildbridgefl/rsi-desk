"""
Strategy contract.

A strategy is a function that takes a price DataFrame plus params and
returns that DataFrame with four boolean columns added:

    long_entry, short_entry, long_exit, short_exit

The engine handles everything else — position tracking, next-bar fills,
costs, stops, max-hold. You only describe the RULE.

To add a rule: copy strategies/_template.py, edit it, import it in
strategies/__init__.py. It shows up in the app automatically.
"""

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

REGISTRY: dict[str, "Strategy"] = {}


@dataclass
class Strategy:
    name: str
    fn: Callable[..., pd.DataFrame]
    params: dict = field(default_factory=dict)
    description: str = ""

    def __call__(self, df: pd.DataFrame, **kw) -> pd.DataFrame:
        p = {**self.params, **kw}
        out = self.fn(df.copy(), **p)
        missing = {"long_entry", "short_entry", "long_exit", "short_exit"} - set(out.columns)
        if missing:
            raise ValueError(f"{self.name} did not set: {sorted(missing)}")
        return out


def register(name: str, params: dict | None = None, description: str = ""):
    """Decorator that adds a strategy to the registry."""
    def deco(fn):
        REGISTRY[name] = Strategy(name=name, fn=fn,
                                  params=params or {},
                                  description=description or (fn.__doc__ or "").strip())
        return fn
    return deco


def get(name: str) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy {name!r}. have: {list(REGISTRY)}")
    return REGISTRY[name]


def names() -> list[str]:
    return list(REGISTRY)
