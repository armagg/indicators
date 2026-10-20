"""The market tier list -- this project's own opinion about which markets belong in production."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from decouple import config as env

from .errors import ParamError, SourceError

MAJOR = "major"
OTHER = "other"
TIERS = (MAJOR, OTHER)

DEFAULT_TIER = OTHER


def normalize_tier(value: str) -> str:
    tier = str(value).strip().lower()
    if tier not in TIERS:
        raise ParamError(f"unknown tier {value!r}; valid: {', '.join(TIERS)}")
    return tier


@dataclass(frozen=True)
class TierStore:
    """Persistence for the tier list. Inject a fake collection in tests."""

    collection: Any

    def ensure_indexes(self) -> None:
        self.collection.create_index(
            [("source", 1), ("market", 1)], unique=True, name="ta_tier_key"
        )

    def get(self, source: str, market: str) -> str:
        doc = self.collection.find_one({"source": source, "market": market})
        return doc["tier"] if doc else DEFAULT_TIER

    def mapping(self, source: str | None = None) -> dict[tuple[str, str], str]:
        """`(source, market) -> tier` for everything classified so far."""
        query = {"source": source} if source else {}
        return {
            (doc["source"], doc["market"]): doc.get("tier", DEFAULT_TIER)
            for doc in self.collection.find(query)
        }

    def list_all(self, source: str | None = None) -> list[dict]:
        query = {"source": source} if source else {}
        return sorted(
            self.collection.find(query),
            key=lambda d: (d.get("source", ""), d.get("market", "")),
        )

    def set(
        self, source: str, market: str, tier: str, *, by: str, note: str = ""
    ) -> dict:
        """Classify a market. `by` is recorded -- tiering is a judgement, and judgements have authors."""
        if not by or not str(by).strip():
            raise ParamError("setting a tier requires an identity")
        doc = {
            "source": source,
            "market": market,
            "tier": normalize_tier(tier),
            "note": note,
            "updated_by": str(by).strip(),
            "updated_at": datetime.now(timezone.utc),
        }
        self.collection.replace_one(
            {"source": source, "market": market}, doc, upsert=True
        )
        return doc

    def clear(self, source: str, market: str) -> bool:
        """Drop an explicit classification, returning the market to `other`."""
        result = self.collection.delete_one({"source": source, "market": market})
        return bool(getattr(result, "deleted_count", 0))


_default: TierStore | None = None


def default_store() -> TierStore:
    global _default
    if _default is None:
        try:
            from pymongo import MongoClient

            client = MongoClient(env("MONGO_URI"), serverSelectionTimeoutMS=5000)
            db = client[env("DB_NAME")]
            store = TierStore(db[env("TIERS_COLLECTION_NAME", default="ta_market_tiers")])
            store.ensure_indexes()
        except Exception as exc:
            raise SourceError(f"cannot reach the tier store: {exc}") from exc
        _default = store
    return _default
