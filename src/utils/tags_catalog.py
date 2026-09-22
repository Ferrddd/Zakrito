

from __future__ import annotations

from pathlib import Path

from src.utils.config import load_config


def load_tags_catalog(path: str | Path = "config/controllable_tags.yaml") -> dict:
    return load_config(path)


def get_column_suffix(catalog: dict, installation: str) -> str:
    try:
        return catalog["installations"][installation]["column_suffix"]
    except KeyError as e:
        raise KeyError(f"Установка '{installation}' не описана в "
                       f"controllable_tags.yaml (секция installations)") from e


def resolve_column(catalog: dict, installation: str, tag_id: str) -> str:
    """(installation, tag_id) -> имя колонки после merge в loader.py
    (напр. ("242000", "T5") -> "T5_hdt", ("avt", "T6") -> "T6_avt")."""
    suffix = get_column_suffix(catalog, installation)
    return f"{tag_id}_{suffix}"