"""Exception types for the TA package; everything deliberate derives from `TAError`."""


class TAError(Exception):
    """Base class for all deliberate errors raised by `ta`."""


class UnknownIndicator(TAError):
    """Requested indicator is not in the registry."""


class ParamError(TAError):
    """A parameter was missing, the wrong type, or out of range."""


class FrameError(TAError):
    """The input OHLCV frame is unusable (missing columns, empty, unsortable)."""


class InsufficientData(TAError):
    """The frame has fewer bars than the indicator's warmup requires."""


class NonCausalIndicator(TAError):
    """A forward-looking indicator was requested on a causal-only path."""


class SourceError(TAError):
    """A data source could not be reached or returned something unusable."""
