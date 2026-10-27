"""Общие фикстуры и эталонные расчёты для проверок.

Эталонные значения считаются здесь же — из выданного CSV-файла, по той же
схеме очистки, которая описана в README. Поэтому тесты не содержат зашитых
«правильных» g и a: они проверяют ваши формулы и качество очистки.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import lab

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "measurements.csv"
G_TABLE = 9.81

EXPECTED_ROWS = 7000
EXPECTED_ANGLES = [8.0, 10.0, 12.0, 15.0, 18.0, 22.0, 26.0]

NUMERIC_COLUMNS = ["angle_deg", "trial", "distance_m", "time_s"]


def reference_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Независимая эталонная очистка данных.

    Правила совпадают с описанными в README: числовой тип, удаление пропусков,
    полных дубликатов, физически невозможных значений и грубых выбросов
    времени по медиане и MAD отдельно для каждого угла.
    """
    data = df.copy()
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["angle_deg", "distance_m", "time_s"])
    data = data.drop_duplicates()
    data = data[data["distance_m"].between(0.5, 1.1)]
    data = data[data["time_s"].between(0.3, 2.0)]
    data = data[data["angle_deg"].between(5.0, 35.0)]

    median = data.groupby("angle_deg")["time_s"].transform("median")
    mad = data.groupby("angle_deg")["time_s"].transform(
        lambda values: (values - values.median()).abs().median()
    )
    modified_z = 0.6745 * (data["time_s"] - median) / mad
    data = data[modified_z.abs() <= 3.5]

    return data.reset_index(drop=True)


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    """Исходный CSV, прочитанный независимо от кода студента."""
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="session")
def loaded() -> pd.DataFrame:
    """Результат вашей функции load_data (журнал без очистки)."""
    return lab.load_data(DATA_PATH)


@pytest.fixture(scope="session")
def clean(loaded: pd.DataFrame) -> pd.DataFrame:
    """Результат вашей функции clean_data."""
    return lab.clean_data(loaded)


@pytest.fixture(scope="session")
def summary(clean: pd.DataFrame) -> pd.DataFrame:
    """Результат вашей функции summarize_by_angle."""
    return lab.summarize_by_angle(clean)


@pytest.fixture(scope="session")
def ref(raw: pd.DataFrame) -> SimpleNamespace:
    """Независимый эталонный расчёт по очищенным данным."""
    table = reference_clean(raw)
    table["a"] = 2.0 * table["distance_m"] / table["time_s"] ** 2

    grouped = table.groupby("angle_deg", sort=True)["a"]
    stats = pd.DataFrame(
        {
            "angle_deg": grouped.mean().index.to_numpy(dtype=float),
            "n": grouped.size().to_numpy(dtype=int),
            "a_mean": grouped.mean().to_numpy(dtype=float),
            "a_std": grouped.std(ddof=1).to_numpy(dtype=float),
        }
    )
    stats["a_sem"] = stats["a_std"] / np.sqrt(stats["n"])
    stats["g_angle"] = stats["a_mean"] / np.sin(np.deg2rad(stats["angle_deg"]))
    stats = stats[["angle_deg", "n", "a_mean", "a_std", "a_sem", "g_angle"]]

    x = np.sin(np.deg2rad(stats["angle_deg"].to_numpy(dtype=float)))
    y = stats["a_mean"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (slope * x + intercept)
    dof = len(x) - 2
    sigma_slope = float(
        np.sqrt(np.sum(residuals**2) / dof / np.sum((x - x.mean()) ** 2))
    )

    return SimpleNamespace(
        frame=table,
        summary=stats,
        g=float(slope),
        g_err=sigma_slope,
        intercept=float(intercept),
    )
