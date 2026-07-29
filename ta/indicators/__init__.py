"""Indicator implementations, grouped by category.

Importing this package registers every indicator with `ta.registry`. That is
the only side effect, and it is the intended one: `ta/__init__.py` imports
this so that `ta.compute(...)` and the HTTP catalogue see the full library
without any caller having to know which module holds which indicator.
"""

from . import momentum, trend, volatility, volume  # noqa: F401
