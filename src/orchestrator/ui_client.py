"""Отправка результата цикла в UI (POST JSON, по умолчанию localhost:3000).

Падение или недоступность UI НЕ должны ломать цикл принятия решения: ошибки
логируются, цикл идёт дальше. Для отладки без UI — dump_path (JSON Lines).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from src.orchestrator.contracts import CycleContext
from src.orchestrator.ui_payload import TagDirectory, TrendBuffer, build_payload

logger = logging.getLogger(__name__)

# ДОПУЩЕНИЕ: путь эндпоинта не известен (в примере указан только порт) — задать через --ui-url / UI_URL.
DEFAULT_UI_URL = "http://localhost:3000/api/update"


class UIPublisher:
    def __init__(self, url: str | None = None, timeout_s: float = 3.0, dump_path: str | Path | None = None,
                 directory: TagDirectory | None = None, tz: str | None = None,
                 trend_horizon_min: int = 60, send: bool = True):
        self.url = url or os.environ.get("UI_URL", DEFAULT_UI_URL)
        self.timeout_s = timeout_s
        self.dump_path = Path(dump_path) if dump_path else None
        self.directory = directory
        self.tz = ZoneInfo(tz) if tz else None
        self.send = send
        self.trend = TrendBuffer(horizon_min=trend_horizon_min)
        self.sent_ok = 0
        self.sent_failed = 0

    def __call__(self, ctx: CycleContext) -> None:
        payload = build_payload(ctx, self.trend, self.directory, self.tz)
        if self.dump_path:
            self.dump_path.parent.mkdir(parents=True, exist_ok=True)
            with self.dump_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        if self.send:
            self._post(payload)

    def _post(self, payload: dict) -> None:
        import httpx
        try:
            r = httpx.post(self.url, json=payload, timeout=self.timeout_s)
            r.raise_for_status()
            self.sent_ok += 1
        except httpx.HTTPError as e:
            self.sent_failed += 1
            logger.warning("UI недоступен или отклонил запрос (%s): %s", self.url, e)