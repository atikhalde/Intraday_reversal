from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


class SignalState:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        try:
            payload: Any = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self.data = {str(key): str(value) for key, value in payload.items()}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self.data = {}

    def contains(self, key: str) -> bool:
        return key in self.data

    def add(self, key: str, timestamp: datetime) -> None:
        self.data[key] = timestamp.isoformat()
        self._prune()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")
        temp.replace(self.path)

    def _prune(self) -> None:
        cutoff = datetime.now(UTC) - timedelta(days=7)
        retained: dict[str, str] = {}
        for key, value in self.data.items():
            try:
                parsed = datetime.fromisoformat(value)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                if parsed.astimezone(UTC) >= cutoff:
                    retained[key] = value
            except ValueError:
                continue
        self.data = retained
