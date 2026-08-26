"""HTTP client for `config-manager`.

Reading is free; exactly one function here writes (`put_market`), and only
`ta/proposals/apply.py` is allowed to call it.

The document shape is the part worth knowing before touching this file,
because the read and write shapes are **not** the same:

    GET /config/market/{source}/{market}
      -> {"source": ..., "market_name": ..., "version": 12,
          "timestamp": ..., "config": { ...the actual fields... }}

    PUT /config/market/{source}/{market}
      <- { ...the actual fields... }        # the INNER object, unwrapped

So a read gives you a wrapper carrying the version, and a write takes the
bare config. PUTting back what you read would nest a wrapper inside a config
and quietly corrupt the document. `market_maker/config_client.py:137` does
the same unwrap (`doc.get("config", {})`), which is the confirmation that
this is the real contract and not an accident of one endpoint.

Two more facts that shape the code:

- Every PUT creates a **new version** (`config_manager.py:142`). There is no
  patch endpoint, so changing one field is a read-modify-write, and the
  version is the only thing that tells you whether someone else wrote in
  between. That check lives in `apply.py`.
- `upsert_config` injects `version` and `timestamp` *into* the inner config
  before storing it. They are the server's bookkeeping, not settings, so
  they are stripped on read and never sent back.

Transport is injectable so tests never touch a socket. The default uses
`urllib` from the standard library rather than `requests`/`httpx`: this is
a handful of JSON calls, and the project has twice been bitten by
third-party wheels not existing for its Python version.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping

from decouple import config as env

from .errors import SourceError

#: Keys config-manager owns and injects into the stored config. Stripped on
#: read; never included in a write.
SERVER_OWNED = frozenset({"version", "timestamp", "_id"})
RISK_OWNED = frozenset({
    "disable_buy", "disable_sell", "sell_to_l0", "buy_size_scale",
})

DEFAULT_TIMEOUT = 10.0


@dataclass(frozen=True)
class MarketConfig:
    """One market's active config, with the version it was read at."""

    source: str
    market: str
    version: int | None
    #: The settings themselves, server bookkeeping already removed.
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wrapper(cls, doc: Mapping[str, Any]) -> "MarketConfig":
        inner = dict(doc.get("config") or {})
        for key in SERVER_OWNED:
            inner.pop(key, None)
        return cls(
            source=doc.get("source", ""),
            market=doc.get("market_name", ""),
            version=doc.get("version"),
            config=inner,
        )

    @property
    def is_enabled(self) -> bool:
        return bool(self.config.get("is_enable"))


class UrllibTransport:
    """Minimal JSON transport. Returns `(status, parsed_body)`.

    HTTP error statuses are returned rather than raised, so callers can tell
    "no config for this market" (404) apart from "config-manager is down"
    (which does raise, as a `SourceError`).
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout

    def request(self, method: str, url: str, body: Any = None) -> tuple[int, Any]:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read().decode()
                return response.status, (json.loads(payload) if payload else None)
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode()
            try:
                parsed = json.loads(payload) if payload else None
            except json.JSONDecodeError:
                parsed = {"detail": payload}
            return exc.code, parsed
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SourceError(f"cannot reach config-manager at {url}: {exc}") from exc


class ConfigManagerClient:
    """Typed wrapper over the config-manager REST surface."""

    def __init__(self, base_url: str | None = None, transport: Any = None):
        self.base_url = (base_url or env("CONFIG_MANAGER_URL")).rstrip("/")
        self.transport = transport or UrllibTransport()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def _url(self, *parts: str, trailing_slash: bool = False) -> str:
        path = "/".join(urllib.parse.quote(str(p), safe="") for p in parts)
        return f"{self.base_url}/config/market/{path}" + ("/" if trailing_slash else "")

    def get_market(self, source: str, market: str) -> MarketConfig | None:
        """Active config for one market, or None when it has none yet."""
        status, body = self.transport.request("GET", self._url(source, market))
        if status == 404:
            return None
        if status >= 400 or not isinstance(body, Mapping):
            raise SourceError(
                f"config-manager returned {status} for {source}/{market}: {body}"
            )
        return MarketConfig.from_wrapper(body)

    def list_markets(self, source: str) -> list[MarketConfig]:
        """Every active config on one exchange. Empty list when none exist."""
        status, body = self.transport.request(
            "GET", self._url(source, trailing_slash=True)
        )
        if status >= 400:
            raise SourceError(f"config-manager returned {status} for {source}: {body}")
        return [MarketConfig.from_wrapper(doc) for doc in (body or [])]

    def versions(self, source: str, market: str) -> list[dict]:
        """Version history, newest first. Empty when the market is unknown."""
        status, body = self.transport.request(
            "GET", f"{self._url(source, market)}/versions"
        )
        if status == 404:
            return []
        if status >= 400:
            raise SourceError(
                f"config-manager returned {status} for {source}/{market} versions: {body}"
            )
        return list(body or [])

    # ------------------------------------------------------------------
    # The single write
    # ------------------------------------------------------------------

    def put_market(
        self, source: str, market: str, settings: Mapping[str, Any]
    ) -> MarketConfig:
        """Replace a market's config, creating a new version.

        `settings` is the **inner** config object, whole. There is no patch
        endpoint: anything omitted here falls back to `ConfigSchema`'s
        defaults rather than keeping its current value, which is why callers
        must compose onto a freshly-read config rather than sending a subset.
        """
        payload = {
            k: v for k, v in settings.items()
            if k not in SERVER_OWNED and k not in RISK_OWNED
        }
        status, body = self.transport.request(
            "PUT", self._url(source, market), body=payload
        )
        if status >= 400 or not isinstance(body, Mapping):
            raise SourceError(
                f"config-manager rejected the write for {source}/{market} "
                f"({status}): {body}"
            )
        return MarketConfig.from_wrapper(body)


def default_client() -> ConfigManagerClient:
    """Env-configured client. `CONFIG_MANAGER_URL` is the service root, e.g.
    `http://localhost:8000` -- the `/config/market/...` path is added here."""
    try:
        return ConfigManagerClient()
    except Exception as exc:
        raise SourceError(f"cannot configure the config-manager client: {exc}") from exc
