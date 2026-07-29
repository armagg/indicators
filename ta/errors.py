"""Exception types for the TA package.

Everything raised deliberately by `ta` derives from `TAError` so the HTTP layer
can map "the caller asked for something impossible" (400) apart from "we broke"
(500). The old service turned every failure into a 500 with a bare `str(e)`.
"""


class TAError(Exception):
    """Base class for all deliberate errors raised by `ta`."""


class UnknownIndicator(TAError):
    """Requested indicator is not in the registry."""


class ParamError(TAError):
    """A parameter was missing, the wrong type, or out of range."""


class FrameError(TAError):
    """The input OHLCV frame is unusable (missing columns, empty, unsortable)."""


class InsufficientData(TAError):
    """The frame has fewer bars than the indicator's warmup requires.

    Raised only when the caller asks for strict warmup handling. The default is
    to compute anyway and report the shortfall in the result metadata, because
    a cold market with 30 bars is a normal state for this desk, not an error.
    """


class NonCausalIndicator(TAError):
    """A forward-looking indicator was requested on a causal-only path.

    See `ta.registry.IndicatorSpec.causal`. Ichimoku's chikou span is the
    canonical example: it is `close` shifted *backwards*, so its value at bar
    `t` is drawn from bar `t + displacement`. Fine on a chart, fatal in a
    backtest.
    """


class SourceError(TAError):
    """A data source could not be reached or returned something unusable."""
