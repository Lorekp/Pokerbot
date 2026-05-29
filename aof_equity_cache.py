"""LRU cache for hero vs range equity (cross-hand reuse in RL training)."""

from __future__ import annotations

from collections import OrderedDict


class EquityCache:
  """Thread-safe enough for single-process training; keyed by coarse sample bucket."""

  def __init__(self, max_entries: int = 4096) -> None:
      self._max = max_entries
      self._data: OrderedDict[tuple, float] = OrderedDict()

  def clear(self) -> None:
      self._data.clear()

  def get(
      self,
      hero_key: str,
      range_str: str,
      samples: int,
  ) -> float | None:
      key = (hero_key, range_str, samples // 50)
      return self._data.get(key)

  def put(
      self,
      hero_key: str,
      range_str: str,
      samples: int,
      value: float,
  ) -> None:
      key = (hero_key, range_str, samples // 50)
      if key in self._data:
          self._data.move_to_end(key)
      self._data[key] = value
      while len(self._data) > self._max:
          self._data.popitem(last=False)


_GLOBAL_EQUITY_CACHE = EquityCache()


def get_equity_cache() -> EquityCache:
    return _GLOBAL_EQUITY_CACHE
