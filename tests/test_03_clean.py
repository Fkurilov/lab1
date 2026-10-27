"""Проверки предобработки журнала (задание 4.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import EXPECTED_ANGLES, EXPECTED_ROWS
import lab

REQUIRED = ["angle_deg", "trial", "distance_m", "time_s"]
CRITICAL = ["angle_deg", "distance_m", "time_s"]


def test_clean_returns_dataframe(clean):
    assert isinstance(clean, pd.DataFrame), (
        "clean_data должна возвращать pandas.DataFrame."
    )


def test_clean_has_required_columns(clean):
    missing = [name for name in REQUIRED if name not in clean.columns]
    assert not missing, f"После очистки пропали колонки: {', '.join(missing)}."


def test_clean_columns_are_numeric(clean):
    for column in REQUIRED:
        assert pd.api.types.is_numeric_dtype(clean[column]), (
            f"Колонка {column} должна быть числовой. "
            "Приведите данные к числу до фильтрации и группировки."
        )


def test_clean_has_no_missing_values(clean):
    for column in CRITICAL:
        count = int(clean[column].isna().sum())
        assert count == 0, (
            f"В колонке {column} осталось {count} пропусков. "
            "Строки без угла, пути или времени нужно удалить "
            "(или обоснованно заполнить, если это возможно)."
        )


def test_clean_has_no_duplicates(clean):
    duplicates = int(clean.duplicated().sum())
    assert duplicates == 0, (
        f"В очищенной таблице осталось {duplicates} полных дубликатов. "
        "Используйте drop_duplicates()."
    )


def test_clean_keeps_only_expected_angles(clean):
    angles = np.sort(clean["angle_deg"].unique().astype(float))
    assert np.allclose(angles, EXPECTED_ANGLES), (
        f"После очистки остались углы {angles.tolist()}, "
        f"а ожидались {EXPECTED_ANGLES}. "
        "Угол нельзя восстанавливать по среднему: строки с пропущенным "
        "углом нужно удалить."
    )


def test_clean_physical_ranges(clean):
    checks = [
        ("angle_deg", 5.0, 35.0, "градусы"),
        ("distance_m", 0.5, 1.1, "метры"),
        ("time_s", 0.3, 2.0, "секунды"),
    ]
    for column, low, high, units in checks:
        values = clean[column].to_numpy(dtype=float)
        assert values.min() >= low and values.max() <= high, (
            f"В колонке {column} остались значения вне разумного диапазона "
            f"[{low}; {high}] {units}: min={values.min():g}, max={values.max():g}. "
            "Проверьте ошибки единиц и сбои датчиков."
        )


def test_clean_keeps_enough_data(clean):
    assert len(clean) < EXPECTED_ROWS, (
        "Похоже, очистка ничего не удалила: в журнале были пропуски, "
        "дубликаты и сбои. Проверьте порядок шагов."
    )
    assert len(clean) >= 6500, (
        f"После очистки осталось всего {len(clean)} строк. "
        "Не удаляйте больше данных, чем нужно: потеряется точность."
    )

    counts = clean.groupby("angle_deg").size()
    assert (counts >= 900).all(), (
        "В одной из серий осталось меньше 900 измерений: "
        f"{counts.to_dict()}. Проверьте критерий выбросов — он не должен "
        "отбрасывать несколько процентов нормальных точек."
    )


def test_clean_does_not_modify_input(raw):
    source = raw.copy(deep=True)
    before = source.copy(deep=True)
    lab.clean_data(source)
    pd.testing.assert_frame_equal(
        source,
        before,
        obj="Исходный журнал",
    )
