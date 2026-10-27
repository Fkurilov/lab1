"""Проверки сводной таблицы по углам (задание 4.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lab

REQUIRED = ["angle_deg", "n", "a_mean", "a_std", "a_sem", "g_angle"]


def _merged(summary, ref, column):
    left = summary[["angle_deg", column]].copy()
    left["angle_deg"] = left["angle_deg"].astype(float)
    merged = left.merge(
        ref.summary[["angle_deg", column]],
        on="angle_deg",
        how="outer",
        suffixes=("_got", "_ref"),
        indicator=True,
    )
    assert (merged["_merge"] == "both").all(), (
        "Набор углов в сводной таблице не совпадает с ожидаемым: "
        f"{merged.loc[merged['_merge'] != 'both', 'angle_deg'].tolist()}"
    )
    return merged.sort_values("angle_deg")


def test_summary_has_required_columns(summary):
    missing = [name for name in REQUIRED if name not in summary.columns]
    assert not missing, (
        "В сводной таблице нет колонок: "
        f"{', '.join(missing)}. Ожидались: {', '.join(REQUIRED)}"
    )


def test_summary_has_one_row_per_angle(summary, ref):
    assert len(summary) == len(ref.summary), (
        f"В сводке {len(summary)} строк вместо {len(ref.summary)} — "
        "должна быть ровно одна строка на каждый угол."
    )
    assert summary["angle_deg"].is_monotonic_increasing, (
        "Строки сводной таблицы должны быть отсортированы по возрастанию angle_deg."
    )


def test_summary_uses_all_clean_rows(summary, clean):
    counts = clean.groupby("angle_deg").size().sort_index()
    got = summary.sort_values("angle_deg")["n"].to_numpy(dtype=int)
    assert np.array_equal(got, counts.to_numpy(dtype=int)), (
        "Колонка n не совпадает с числом измерений после очистки: "
        f"в сводке {got.tolist()}, в очищенных данных {counts.tolist()}."
    )


def test_summary_mean_acceleration(summary, ref, clean):
    merged = _merged(summary, ref, "a_mean")
    for row in merged.itertuples(index=False):
        got = float(row.a_mean_got)
        expected = float(row.a_mean_ref)
        if not np.isclose(got, expected, rtol=5e-3, atol=1e-6):
            pytest.fail(
                f"Для угла {row.angle_deg:g}° среднее ускорение {got:.6f} м/с², "
                f"а после корректной очистки получается {expected:.6f} м/с² "
                f"(разница {abs(got - expected):.2e}).\n"
                "Проверьте: (1) формулу a = 2·s/t², (2) что усредняются все "
                "оставшиеся попытки, (3) что грубые выбросы удалены, "
                "(4) что промежуточные значения не округлены слишком рано."
            )

    # Сводка должна быть построена именно по очищенной таблице.
    local = 2.0 * clean["distance_m"] / clean["time_s"] ** 2
    expected_local = local.groupby(clean["angle_deg"]).mean().sort_index()
    got_local = summary.sort_values("angle_deg")["a_mean"].to_numpy(dtype=float)
    assert np.allclose(got_local, expected_local.to_numpy(), rtol=1e-6), (
        "a_mean не совпадает со средним по вашей же очищенной таблице. "
        "Проверьте, что summarize_by_angle вызывается после clean_data."
    )


def test_summary_g_per_angle(summary, ref):
    merged = _merged(summary, ref, "g_angle")
    for row in merged.itertuples(index=False):
        got = float(row.g_angle_got)
        expected = float(row.g_angle_ref)
        if not np.isclose(got, expected, rtol=5e-3, atol=1e-6):
            pytest.fail(
                f"Для угла {row.angle_deg:g}° оценка g = {got:.6f} м/с², "
                f"а ожидалось {expected:.6f} м/с².\n"
                "Проверьте формулу g = a_mean / sin(alpha) и перевод угла "
                "из градусов в радианы (np.deg2rad)."
            )


def test_summary_scatter_is_small_after_cleaning(summary):
    ratio = summary["a_std"].to_numpy(dtype=float) / summary["a_mean"].to_numpy(dtype=float)
    assert np.all(ratio > 0), "a_std должно быть положительным."
    assert np.all(ratio < 0.08), (
        "Разброс ускорения внутри серии слишком велик: "
        f"a_std/a_mean достигает {ratio.max():.3f} (это {ratio.max() * 100:.1f}%). "
        "Похоже, грубые выбросы времени не удалены. "
        "После очистки относительный разброс должен быть порядка 2–4%."
    )


def test_summary_std_matches_reference(summary, ref):
    merged = _merged(summary, ref, "a_std")
    for row in merged.itertuples(index=False):
        got = float(row.a_std_got)
        expected = float(row.a_std_ref)
        assert np.isfinite(got) and got > 0, (
            f"Для угла {row.angle_deg:g}° СКО должно быть положительным числом."
        )
        assert abs(got - expected) <= 0.30 * expected, (
            f"Для угла {row.angle_deg:g}° a_std = {got:.6f}, "
            f"а после корректной очистки {expected:.6f}.\n"
            "Используйте выборочное СКО (ddof=1) и не удаляйте лишние точки."
        )


def test_summary_sem_consistent_with_std(summary):
    n = summary["n"].to_numpy(dtype=float)
    expected = summary["a_std"].to_numpy(dtype=float) / np.sqrt(n)
    got = summary["a_sem"].to_numpy(dtype=float)
    assert np.allclose(got, expected, rtol=0.06, atol=1e-12), (
        "a_sem должна быть стандартной ошибкой среднего: "
        "a_sem = a_std / sqrt(n)."
    )


def test_summary_does_not_modify_input(ref):
    df = ref.frame.copy()
    before = df.copy(deep=True)
    lab.summarize_by_angle(df)
    pd.testing.assert_frame_equal(df, before, obj="Входная таблица")
