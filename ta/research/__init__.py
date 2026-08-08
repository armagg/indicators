"""Research: signal families, the brute force, saved scenarios, and the queue.

The service surface is in `app.py` (`/research/*`); the CLI is
`research/signal_brute.py`. Both go through `run_brute`, so a result never
depends on how it was launched.
"""

from .brute import JOB_NAME, run_brute  # noqa: F401
from .families import FAMILIES, VOLUME_CAVEAT  # noqa: F401
from .jobs import JobQueue, default_queue, job_id  # noqa: F401
from .scenarios import DEFAULTS, SCENARIOS, expand, list_scenarios  # noqa: F401
