"""Прогон оркестратора по историческому периоду (демо/бэктест).

    python -m scripts.run_orchestrator --start 2026-06-01 --steps 12
    python -m scripts.run_orchestrator --start 2026-06-01 --steps 200 --every 6 --remote http://localhost:8001

Берёт строки из data/converted/telemetry_pac_lims.parquet, «проигрывает» их как поток
состояний (агент сам копит историю), печатает рекомендацию и пишет аудит-лог в
data/converted/audit/cycles.jsonl.

Агенты надёжности и оптимизации строятся по истории ДО --start (модельные границы
управляющих тегов, базовая линия «нормы»), поэтому будущее в них не просачивается.
Опорные значения модельных диапазонов сохраняются в data/converted/audit/control_bounds.json.
С флагом --stubs надёжность и оптимизатор заменяются заглушками (как раньше).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd

from src.agents.base import Agent
from src.agents.optimization import OptimizationAgent, build_reference, tag_directory
from src.agents.quality.agent import QualityAgent
from src.agents.reliability import ReliabilityAgent
from src.data_pipeline.pipeline_models import data_pipeline_settings
from src.orchestrator.adapters import state_from_row
from src.orchestrator.clients import RemoteQualityAgent
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.ui_client import UIPublisher
from src.utils.config import setup_logging


def main() -> None:
    setup_logging()
    logging.getLogger("src.data_pipeline.features").setLevel(logging.WARNING)  # add_engineered шумит на каждом цикле
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/agents/quality/config.yaml")
    ap.add_argument("--start", required=True, help="время первого состояния, напр. 2026-06-01 08:00")
    ap.add_argument("--steps", type=int, default=12, help="сколько циклов проиграть")
    ap.add_argument("--every", type=int, default=1, help="шаг по сетке (1 = каждые 10 мин)")
    ap.add_argument("--warmup", type=int, default=288, help="точек истории до --start для прогрева буфера")
    ap.add_argument("--remote", default=None, help="URL HTTP-агента качества вместо in-process")
    ap.add_argument("--audit", default="data/converted/audit/cycles.jsonl")
    ap.add_argument("--ui-url", default=None,
                    help="POST результата каждого цикла в UI, напр. http://localhost:3000/api/cycles "
                         "(или переменная UI_URL). Без флага в UI ничего не отправляется")
    ap.add_argument("--ui-dump", default=None, help="писать JSON для UI в файл (JSON Lines), без отправки")
    ap.add_argument("--ui-tz", default=None, help="таймзона для меток времени, напр. Europe/Moscow")
    ap.add_argument("--sleep", type=float, default=0.0, help="пауза между циклами, сек (для демо с UI)")
    ap.add_argument("--stubs", action="store_true", help="заглушки вместо агентов надёжности и оптимизации")
    ap.add_argument("--bounds-out", default="data/converted/audit/control_bounds.json",
                    help="куда сохранить опорные значения модельных диапазонов (для аудита)")
    args = ap.parse_args()

    df = pd.read_parquet(data_pipeline_settings.converted_data_path / "telemetry_pac_lims.parquet")
    start = pd.Timestamp(args.start)
    pos = int(df.index.searchsorted(start))
    if pos >= len(df):
        raise SystemExit(f"--start {start} за пределами данных ({df.index.min()} .. {df.index.max()})")

    quality: Agent
    if args.remote:
        remote = RemoteQualityAgent(args.remote)
        remote.warm_up(df.iloc[max(0, pos - args.warmup):pos])   # POST /history одним-двумя батчами
        quality = remote
    else:
        local = QualityAgent.load(args.config)
        local.warm_up(df.iloc[max(0, pos - args.warmup):pos])
        quality = local

    reliability = optimizer = None
    directory = None
    if not args.stubs:
        history = df.iloc[:pos]                      # строго до --start: без утечки из будущего
        if history.empty:
            raise SystemExit("Нет истории до --start — нечем строить границы и базовую линию")
        reference = build_reference(history)          # опорные значения для модельных диапазонов
        Path(args.bounds_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.bounds_out).write_text(json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8")
        directory = tag_directory(df.columns)
        reliability = ReliabilityAgent.fit(history)
        reliability.warm_up(df.iloc[max(0, pos - args.warmup):pos])
        optimizer = OptimizationAgent(reference=reference)

    publisher = None
    ui_url = args.ui_url or os.environ.get("UI_URL")
    if ui_url or args.ui_dump:
        publisher = UIPublisher(url=ui_url, dump_path=args.ui_dump, directory=directory,
                                tz=args.ui_tz, send=bool(ui_url))
    orch = Orchestrator(quality, reliability=reliability, optimizer=optimizer,
                        audit_path=args.audit, publisher=publisher)
    stubs = [a.name for a in (orch.reliability, orch.optimizer) if getattr(a, "is_stub", False)]
    if stubs:
        logging.getLogger("run_orchestrator").warning("Заглушки вместо реальных агентов: %s", ", ".join(stubs))
    log = logging.getLogger("run_orchestrator")
    for i in range(args.steps):
        j = pos + i * args.every
        if j >= len(df):
            break
        ts, row = df.index[j], df.iloc[j]
        rec = orch.run_cycle(state_from_row(ts, row))
        log.info("%s | %-18s | %s | conf=%.2f", ts, rec.status.value, rec.headline, rec.confidence)
        if rec.problem:
            log.info("    проблема: %s", rec.problem)
        for a in rec.action:
            log.info("    действие: %s  %s -> %s", a["tag"], a["current"], a["recommended"])
        if rec.explanation:
            log.info("    %s", rec.explanation)
        for w in rec.warnings:
            if not w.startswith("Заглушки"):
                log.info("    ! %s", w)
        if args.sleep:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()