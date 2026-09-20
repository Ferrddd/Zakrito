from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlDefinition:
    tag: str
    description: str
    unit: str
    controllability: str


# Источник: лист КИП из Теги_хакатон.
#
# Важно:
# - "управляемая" -> кандидат на управляющее воздействие;
# - "не управляемая" -> исключаем;
# - "спорно"/"возможно"/"наверное не управляемая" -> исключаем
#   из автоматического пространства действий;
# - пустое значение -> пока не считаем управляемым.
#
# Поле "вывод/берем" намеренно не используется как источник
# controllability.


CONTROL_CATALOG: tuple[ControlDefinition, ...] = (
    # -------------------------
    # АВТ
    # -------------------------

    ControlDefinition(
        tag="T1",
        description="Температура верха К1. Выход на FIRC0955",
        unit="°C",
        controllability="управляемая",
    ),
    ControlDefinition(
        tag="P2",
        description="Давление бензиновых паров верха К1",
        unit="МПа",
        controllability="управляемая",
    ),
    ControlDefinition(
        tag="T6",
        description="Температура низа К1",
        unit="°C",
        controllability="управляемая",
    ),
    ControlDefinition(
        tag="F12",
        description="Расход 2 ЦО в К2",
        unit="т/ч",
        controllability="управляемая",
    ),
    ControlDefinition(
        tag="F14",
        description="Расход 1 ЦО в К2",
        unit="т/ч",
        controllability="управляемая",
    ),
    ControlDefinition(
        tag="F19",
        description="Расход острого орошения К2",
        unit="т/ч",
        controllability="управляемая",
    ),
    ControlDefinition(
        tag="P22",
        description="Давление верха К2",
        unit="МПа",
        controllability="управляемая",
    ),
    ControlDefinition(
        tag="F25",
        description="Расход бензина после Т26",
        unit="т/ч",
        controllability="управляемая",
    ),
    ControlDefinition(
        tag="F30",
        description="Расход фр. 290-350 °С с установки",
        unit="т/ч",
        controllability="управляемая",
    ),
    ControlDefinition(
        tag="F32",
        description="Расход фр. 240-290 °С с установки",
        unit="т/ч",
        controllability="управляемая",
    ),
    ControlDefinition(
        tag="T33",
        description="Температура низа К-2",
        unit="°C",
        controllability="управляемая",
    ),
    ControlDefinition(
        tag="F34",
        description="Расход фр. 150-250 °С с установки",
        unit="т/ч",
        controllability="управляемая",
    ),
    ControlDefinition(
        tag="T37",
        description="Температура ВЦО в К-10",
        unit="°C",
        controllability="управляемая",
    ),
    ControlDefinition(
        tag="T49",
        description="Температура верха К-10",
        unit="°C",
        controllability="управляемая",
    ),
    ControlDefinition(
        tag="F56",
        description="Расход фракции до 350 °С с установки",
        unit="т/ч",
        controllability="управляемая",
    ),
    ControlDefinition(
        tag="F57",
        description="Расход виртуальной фракции до 350 °С с установки",
        unit="т/ч",
        controllability="управляемая",
    ),
    ControlDefinition(
        tag="F59",
        description="Расход фр. 420-500 °С с установки",
        unit="м³/ч",
        controllability="управляемая",
    ),
    ControlDefinition(
        tag="F60",
        description="Расход фр. 350-560 °С с установки",
        unit="т/ч",
        controllability="управляемая",
    ),
    ControlDefinition(
        tag="F64",
        description="Расход 3-го ЦО в К2",
        unit="т/ч",
        controllability="управляемая",
    ),
    ControlDefinition(
        tag="F65",
        description="Производительность К-2 по отбензиненной нефти",
        unit="т/ч",
        controllability="управляемая",
    ),
)