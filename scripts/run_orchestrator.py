"""Прогон оркестратора по историческому периоду (демо/бэктест).

    python -m scripts.run_orchestrator --start 2026-06-01 --steps 12
    python -m scripts.run_orchestrator --start 2026-06-01 --steps 200 --every 6 --remote http://localhost:8001

Берёт строки из data/converted/telemetry_pac_lims.parquet, «проигрывает» их как поток
состояний (агент сам копит историю), печатает рекомендацию и пишет аудит-лог в
data/converted/audit/cycles.jsonl.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.agents.quality.agent import QualityAgent
from src.data_pipeline.pipeline_models import data_pipeline_settings
from src.orchestrator.adapters import state_from_row
from src.orchestrator.clients import RemoteQualityAgent
from src.orchestrator.orchestrator import Orchestrator
from src.utils.config import setup_logging


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/agents/quality/config.yaml")
    ap.add_argument("--start", required=True, help="время первого состояния, напр. 2026-06-01 08:00")
    ap.add_argument("--steps", type=int, default=12, help="сколько циклов проиграть")
    ap.add_argument("--every", type=int, default=1, help="шаг по сетке (1 = каждые 10 мин)")
    ap.add_argument("--warmup", type=int, default=288, help="точек истории до --start для прогрева буфера")
    ap.add_argument("--remote", default=None, help="URL HTTP-агента качества вместо in-process")
    ap.add_argument("--audit", default="data/converted/audit/cycles.jsonl")
    args = ap.parse_args()

    df = pd.read_parquet(data_pipeline_settings.converted_data_path / "telemetry_pac_lims.parquet")
    start = pd.Timestamp(args.start)
    pos = int(df.index.searchsorted(start))
    if pos >= len(df):
        raise SystemExit(f"--start {start} за пределами данных ({df.index.min()} .. {df.index.max()})")

    if args.remote:
        quality = RemoteQualityAgent(args.remote)
        for ts, row in df.iloc[max(0, pos - args.warmup):pos].iterrows():
            quality.evaluate(state_from_row(ts, row))   # прогрев удалённого буфера
    else:
        quality = QualityAgent.load(args.config)
        quality.warm_up(df.iloc[max(0, pos - args.warmup):pos])

    orch = Orchestrator(quality, audit_path=args.audit)
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
        for w in rec.warnings:
            log.info("    ! %s", w)


if __name__ == "__main__":
    main()