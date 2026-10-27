"""Проверки МНК-оценки g (задание 4.5)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import G_TABLE
import lab


def _unpack(result):
    try:
        g, g_err, intercept = result
    except Exception:
        pytest.fail(
            "fit_g должна возвращать кортеж из трёх чисел: "
            "(g, sigma_g, intercept)."
        )
    return float(g), float(g_err), float(intercept)


def test_fit_returns_three_numbers(ref):
    result = lab.fit_g(ref.summary)
    assert hasattr(result, "__len__") and len(result) == 3, (
        "fit_g должна возвращать ровно три значения: "
        "(g, sigma_g, intercept)."
    )
    g, g_err, intercept = _unpack(result)
    assert np.isfinite([g, g_err, intercept]).all(), (
        "Среди возвращаемых значений есть NaN или бесконечность."
    )


def test_fit_slope_matches_reference(ref):
    g, _, _ = _unpack(lab.fit_g(ref.summary))
    assert abs(g - ref.g) <= 0.03, (
        f"Наклон прямой получился {g:.4f} м/с², а по очищенным данным МНК даёт "
        f"{ref.g:.4f} м/с².\n"
        "Проверьте: по оси X откладывается sin(alpha) (безразмерный), "
        "по оси Y — a_mean; np.polyfit возвращает [наклон, свободный член]. "
        "Если наклон сильно занижен, возможно, в данные попали грубые выбросы."
    )


def test_fit_intercept_matches_reference(ref):
    _, _, intercept = _unpack(lab.fit_g(ref.summary))
    assert abs(intercept - ref.intercept) <= 0.02, (
        f"Свободный член прямой получился {intercept:+.4f} м/с², "
        f"а МНК даёт {ref.intercept:+.4f} м/с².\n"
        "Не подгоняйте прямую через начало координат: свободный член "
        "нужен для оценки сопротивления движению. Проверьте также очистку "
        "выбросов: они смещают свободный член."
    )


def test_fit_g_is_close_to_table_value(ref):
    g, _, _ = _unpack(lab.fit_g(ref.summary))
    assert abs(g - G_TABLE) <= 0.1, (
        f"Получено g = {g:.3f} м/с², что далеко от табличного {G_TABLE} м/с².\n"
        "Если значение около 0.17 — по оси X отложен угол в радианах вместо "
        "sin(alpha). Если около 560 — угол в градусах подставлен в sin напрямую. "
        "Если около 9.7 — не удалены грубые выбросы времени."
    )


def test_fit_slope_error_is_plausible(ref):
    _, g_err, _ = _unpack(lab.fit_g(ref.summary))
    assert 0.0005 < g_err < 0.05, (
        f"sigma_g = {g_err:.4g} м/с² выглядит неправдоподобно.\n"
        "Оцените стандартную ошибку наклона по остаткам: "
        "sigma_g = sqrt( sum(r²)/(N−2) / sum((x−mean(x))²) ). "
        "При 1000 измерениях на угол она должна быть порядка 0.01 м/с²."
    )


def test_fit_on_ideal_data_is_exact(ref):
    ideal = ref.summary.copy()
    ideal["a_mean"] = G_TABLE * np.sin(np.deg2rad(ideal["angle_deg"]))
    g, _, intercept = _unpack(lab.fit_g(ideal))
    assert abs(g - G_TABLE) < 1e-6, (
        "На точных данных a = 9.81·sin(alpha) МНК должен дать g = 9.81. "
        f"Получено {g:.6f}."
    )
    assert abs(intercept) < 1e-6, (
        "На точных данных без трения свободный член должен быть равен нулю. "
        f"Получено {intercept:.3e}."
    )


def test_fit_on_student_summary(summary, ref):
    g, _, _ = _unpack(lab.fit_g(summary))
    assert abs(g - ref.g) <= 0.05, (
        f"По вашей сводной таблице МНК даёт g = {g:.4f} м/с², "
        f"а по эталонной очистке {ref.g:.4f} м/с². "
        "Скорее всего, отличается очистка данных или расчёт a_mean."
    )
