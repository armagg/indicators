"""The hourly clock for shadow mode.

A daemon thread inside the service rather than a host cron, so the schedule
ships with the image and cannot drift out of sync with the code that runs
on it. It is deliberately dumb: wake, observe, sleep. There is no catch-up
of missed intervals — a gap in the record is information (the service was
down), and the scorer already refuses to measure a forward return across
one. Backfilling would hide exactly the thing worth seeing.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable

logger = logging.getLogger(__name__)


class ShadowScheduler:
    """Runs `job()` every `interval_sec`, starting after a short delay."""

    def __init__(self, job: Callable[[], dict], interval_sec: float = 3600.0,
                 initial_delay_sec: float = 20.0):
        self.job = job
        self.interval_sec = float(interval_sec)
        self.initial_delay_sec = float(initial_delay_sec)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_run: datetime | None = None
        self.last_result: dict | None = None
        self.last_error: str | None = None
        self.runs = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._loop, name="shadow-scheduler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        if self._stop.wait(self.initial_delay_sec):
            return
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self.interval_sec)

    def run_once(self) -> dict | None:
        """One observation. Never raises -- a failed hour must not kill the loop."""
        try:
            self.last_result = self.job()
            self.last_error = None
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("shadow observation failed: %s", self.last_error)
        self.last_run = datetime.now(timezone.utc)
        self.runs += 1
        return self.last_result

    def status(self) -> dict:
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "interval_sec": self.interval_sec,
            "runs": self.runs,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_error": self.last_error,
            "last_result": self.last_result,
        }


_scheduler: ShadowScheduler | None = None


def default_scheduler() -> ShadowScheduler | None:
    """Env-configured scheduler, or None when no markets are watched."""
    global _scheduler
    if _scheduler is None:
        from decouple import config as env

        from ..configclient import ConfigManagerClient
        from .shadow import default_store, observe_all, watchlist_from_env

        watchlist = watchlist_from_env()
        if not watchlist:
            return None

        def job() -> dict:
            try:
                client = ConfigManagerClient()
            except Exception:
                client = None
            return observe_all(
                watchlist, store=default_store(), config_client=client
            )

        _scheduler = ShadowScheduler(
            job, interval_sec=float(env("SHADOW_INTERVAL_SEC", default=3600))
        )
        _scheduler.start()
    return _scheduler
