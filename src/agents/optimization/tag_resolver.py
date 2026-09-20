# src/agents/optimization/tag_resolver.py

from __future__ import annotations

import re


_SUFFIX_PATTERN = re.compile(r"_[A-Za-z]+$")


def normalize_tag(tag: str) -> str:
    """
    Приводит техническое имя тега к каноническому имени.

    Примеры:
        T6     -> T6
        T6_x   -> T6
        T6_y   -> T6
        F9_x   -> F9
        P24_y  -> P24

    Каноническое имя предполагается таким, как оно указано в КИП.
    """
    return _SUFFIX_PATTERN.sub("", str(tag))


def build_tag_mapping(
    catalog_tags: set[str],
    available_tags: set[str],
) -> dict[str, str]:
    """
    Строит соответствие:

        канонический тег из КИП -> фактический тег в данных

    Приоритет:
    1. точное совпадение;
    2. совпадение после удаления технического суффикса.

    Если для одного канонического тега найдено несколько
    вариантов с суффиксами, выбрасывается ValueError,
    чтобы случайно не выбрать неправильный сигнал.
    """

    mapping: dict[str, str] = {}

    # ---------------------------------------------------------
    # 1. Точные совпадения имеют максимальный приоритет.
    # ---------------------------------------------------------
    for catalog_tag in catalog_tags:
        if catalog_tag in available_tags:
            mapping[catalog_tag] = catalog_tag

    # ---------------------------------------------------------
    # 2. Для остальных ищем варианты с техническим суффиксом.
    # ---------------------------------------------------------
    for catalog_tag in catalog_tags:
        if catalog_tag in mapping:
            continue

        candidates = sorted(
            available_tag
            for available_tag in available_tags
            if normalize_tag(available_tag) == catalog_tag
        )

        if not candidates:
            continue

        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous tag mapping for '{catalog_tag}': "
                f"found multiple candidates {candidates}. "
                "Please resolve the physical tag mapping explicitly."
            )

        mapping[catalog_tag] = candidates[0]

    return mapping


def resolve_tag(
    catalog_tag: str,
    available_tags: set[str],
) -> str | None:
    """
    Находит фактическое имя тега в данных для тега из КИП.

    Примеры:
        resolve_tag("T6", {"T1", "T6_x"}) -> "T6_x"
        resolve_tag("F9", {"F9_y"}) -> "F9_y"
        resolve_tag("T1", {"T1", "T1_x"}) -> "T1"

    Если найдено несколько вариантов с суффиксами,
    выбрасывается ValueError.
    """

    # Сначала точное совпадение.
    if catalog_tag in available_tags:
        return catalog_tag

    # Затем варианты с суффиксом.
    candidates = sorted(
        available_tag
        for available_tag in available_tags
        if normalize_tag(available_tag) == catalog_tag
    )

    if not candidates:
        return None

    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous tag mapping for '{catalog_tag}': "
            f"found multiple candidates {candidates}. "
            "Please resolve the physical tag mapping explicitly."
        )

    return candidates[0]


def validate_control_catalog(
    control_tags: set[str],
    available_tags: set[str],
) -> dict[str, list[str]]:
    """
    Проверяет, какие управляющие теги из каталога
    удалось сопоставить с тегами в historical_stats.csv.

    Возвращает:
        {
            "matched": [...],
            "missing": [...],
            "mapping": {...},
        }
    """

    mapping: dict[str, str] = {}
    missing: list[str] = []

    for catalog_tag in sorted(control_tags):
        resolved = resolve_tag(catalog_tag, available_tags)

        if resolved is None:
            missing.append(catalog_tag)
        else:
            mapping[catalog_tag] = resolved

    return {
        "matched": sorted(mapping.keys()),
        "missing": missing,
        "mapping": mapping,
    }