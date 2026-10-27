"""Проверки чтения журнала измерений (задание 4.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import DATA_PATH, EXPECTED_ANGLES, EXPECTED_ROWS
import lab

REQUIRED = ["angle_deg", "trial", "distance_m", "time_s"]


def test_load_data_returns_dataframe(loaded):
    assert isinstance(loaded, pd.DataFrame), (
        "load_data должна возвращать pandas.DataFrame, "
        f"а вернула {type(loaded).__name__}"
    )


def test_load_data_has_required_columns(loaded):
    missing = [name for name in REQUIRED if name not in loaded.columns]
    assert not missing, (
        "В таблице нет обязательных колонок: "
        f"{', '.join(missing)}. Ожидались: {', '.join(REQUIRED)}"
    )


def test_load_data_keeps_all_raw_rows(loaded, raw):
    assert len(raw) == EXPECTED_ROWS, (
        "Файл data/measurements.csv изменён: в нём должно быть "
        f"{EXPECTED_ROWS} строк. Восстановите его из репозитория."
    )
    assert len(loaded) == EXPECTED_ROWS, (
        f"load_data вернула {len(loaded)} строк вместо {EXPECTED_ROWS}. "
        "На этом этапе очищать данные нельзя: журнал нужно прочитать как есть, "
        "а пропуски/дубликаты обработать в clean_data."
    )


def test_load_data_preserves_missing_values(loaded, raw):
    expected_missing = int(raw.isna().sum().sum())
    got_missing = int(loaded.isna().sum().sum())
    assert expected_missing > 0, "В журнале должны быть пропуски — проверьте файл данных."
    assert got_missing == expected_missing, (
        f"load_data сохранила {got_missing} пропусков вместо {expected_missing}. "
        "Не заполняйте и не удаляйте пропуски при чтении: это задача clean_data."
    )


def test_load_data_preserves_duplicates(loaded, raw):
    expected_duplicates = int(raw.duplicated().sum())
    got_duplicates = int(loaded.duplicated().sum())
    assert expected_duplicates > 0, "В журнале должны быть дубликаты — проверьте файл данных."
    assert got_duplicates == expected_duplicates, (
        f"load_data сохранила {got_duplicates} полных дубликатов вместо "
        f"{expected_duplicates}. Дубликаты нужно удалять в clean_data."
    )


def test_load_data_values_match_file(loaded, raw):
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True),
        raw.reset_index(drop=True),
        check_dtype=False,
        obj="Журнал измерений",
    )


def test_load_data_reads_all_angles(loaded):
    got = np.sort(loaded["angle_deg"].dropna().unique().astype(float))
    assert np.allclose(got, EXPECTED_ANGLES), (
        "Набор углов не совпадает с журналом: "
        f"получено {got.tolist()}, ожидалось {EXPECTED_ANGLES}."
    )


def test_load_data_default_path(loaded):
    default = lab.load_data()
    assert isinstance(default, pd.DataFrame) and len(default) == len(loaded), (
        "load_data() без аргумента должна читать data/measurements.csv. "
        "Проверьте значение DATA_PATH по умолчанию."
    )


def test_load_data_custom_path(tmp_path, raw):
    subset = raw.head(8)
    path = tmp_path / "subset.csv"
    subset.to_csv(path, index=False)

    for argument in (path, str(path)):
        got = lab.load_data(argument)
        assert len(got) == len(subset), (
            "load_data должна принимать путь к произвольному CSV-файлу "
            f"(передан {type(argument).__name__})."
        )
        assert int(got.isna().sum().sum()) == int(subset.isna().sum().sum()), (
            "При чтении другого файла пропуски должны сохраняться."
        )


def test_data_path_points_to_existing_file():
    assert DATA_PATH.exists(), (
        "Не найден data/measurements.csv. Проверьте, что файл данных на месте."
    )
