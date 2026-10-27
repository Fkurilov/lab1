"""Проверки формулы ускорения (задание 4.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lab


@pytest.mark.parametrize(
    ("distance", "time", "expected"),
    [
        (0.800, 0.500, 6.4),
        (1.000, 2.000, 0.5),
        (0.500, 0.250, 16.0),
    ],
)
def test_acceleration_scalar(distance, time, expected):
    values = np.asarray(lab.acceleration(distance, time), dtype=float).reshape(-1)
    assert values.size == 1, (
        "Для скалярных аргументов acceleration должна вернуть одно число."
    )
    got = float(values[0])
    assert got == pytest.approx(expected, rel=1e-9), (
        "Формула ускорения при равноускоренном движении: a = 2·s/t². "
        f"Для s={distance} м, t={time} с ожидалось {expected}, получено {got}."
    )


def test_acceleration_vectorized():
    distance = np.array([0.800, 0.800, 0.500, 1.200])
    time = np.array([0.500, 1.000, 0.250, 0.800])
    expected = 2.0 * distance / time**2

    got = np.asarray(lab.acceleration(distance, time), dtype=float)
    assert got.shape == expected.shape, (
        "acceleration должна работать поэлементно с массивами: "
        f"вернулось {got.shape} значений вместо {expected.shape}."
    )
    assert np.allclose(got, expected, rtol=1e-9), (
        "При расчёте по массивам формула a = 2·s/t² даёт другой результат."
    )


def test_acceleration_accepts_pandas_series():
    distance = pd.Series([0.800, 0.800])
    time = pd.Series([0.500, 0.250])

    got = np.asarray(lab.acceleration(distance, time), dtype=float)
    assert np.allclose(got, [6.4, 25.6], rtol=1e-9), (
        "acceleration должна принимать pandas.Series (так её вызывает main.py)."
    )


def test_acceleration_does_not_mutate_input():
    distance = np.array([0.800, 0.900])
    time = np.array([0.500, 0.600])
    distance_copy = distance.copy()
    time_copy = time.copy()

    lab.acceleration(distance, time)

    assert np.array_equal(distance, distance_copy) and np.array_equal(time, time_copy), (
        "Функция не должна изменять переданные ей массивы."
    )
