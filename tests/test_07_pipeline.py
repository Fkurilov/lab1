"""Проверка полного сценария main.py (задание 4.7)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib import image as mpimg

from conftest import DATA_PATH, EXPECTED_ROWS, G_TABLE, ROOT

REQUIRED_SUMMARY_COLUMNS = ["angle_deg", "n", "a_mean", "a_std", "a_sem", "g_angle"]


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory) -> SimpleNamespace:
    """Копирует проект во временную папку и запускает там python main.py."""
    work = tmp_path_factory.mktemp("pipeline")
    # Копируем все модули проекта: студент может вынести часть кода в отдельный файл.
    for module in ROOT.glob("*.py"):
        shutil.copy2(module, work / module.name)
    shutil.copytree(ROOT / "data", work / "data")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.run(
        [sys.executable, "main.py"],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    return SimpleNamespace(root=work, proc=proc)


def _require_success(pipeline):
    assert pipeline.proc.returncode == 0, (
        "python main.py завершился с ошибкой.\n"
        f"Код возврата: {pipeline.proc.returncode}\n"
        f"stdout:\n{pipeline.proc.stdout}\n"
        f"stderr:\n{pipeline.proc.stderr}"
    )


def test_main_script_runs(pipeline):
    _require_success(pipeline)


def test_cleaning_log_created(pipeline):
    _require_success(pipeline)
    path = pipeline.root / "results" / "cleaning.txt"
    assert path.exists(), "main.py не создал results/cleaning.txt."
    text = path.read_text(encoding="utf-8").lower()
    assert "пропуск" in text and "дубликат" in text, (
        "results/cleaning.txt должен содержать сведения о пропусках и дубликатах."
    )
    assert "после очистки" in text, (
        "results/cleaning.txt должен содержать число строк после очистки."
    )


def test_g_result_json(pipeline, ref):
    _require_success(pipeline)
    path = pipeline.root / "results" / "g_result.json"
    assert path.exists(), (
        "main.py не создал results/g_result.json. "
        "В нём должны быть ключи g, g_err, intercept."
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = {"g", "g_err", "intercept"} - set(payload)
    assert not missing, f"В g_result.json нет ключей: {sorted(missing)}"

    g = float(payload["g"])
    assert abs(g - ref.g) <= 0.03, (
        f"В results/g_result.json записано g = {g:.4f}, "
        f"а МНК по очищенным данным даёт {ref.g:.4f}."
    )
    assert abs(g - G_TABLE) <= 0.1, (
        f"В results/g_result.json g = {g:.4f} м/с² — это далеко от {G_TABLE} м/с²."
    )
    assert float(payload["g_err"]) > 0, "Погрешность g должна быть положительной."


def test_summary_csv(pipeline, ref):
    _require_success(pipeline)
    path = pipeline.root / "results" / "summary.csv"
    assert path.exists(), "main.py не создал results/summary.csv."
    table = pd.read_csv(path)
    missing = [name for name in REQUIRED_SUMMARY_COLUMNS if name not in table.columns]
    assert not missing, f"В results/summary.csv нет колонок: {', '.join(missing)}"
    assert len(table) == len(ref.summary), (
        f"В results/summary.csv {len(table)} строк вместо {len(ref.summary)}."
    )
    assert np.allclose(
        table["a_mean"].to_numpy(dtype=float),
        ref.summary["a_mean"].to_numpy(dtype=float),
        rtol=5e-3,
        atol=1e-6,
    ), "Средние ускорения в results/summary.csv не совпадают с расчётом."


def test_plots_and_report_created(pipeline):
    _require_success(pipeline)
    for relative in ("plots/a_vs_sin_alpha.png", "plots/g_vs_angle.png"):
        path = pipeline.root / relative
        assert path.exists(), f"main.py не создал {relative}."
        image = mpimg.imread(path)
        assert float(np.std(image)) > 0.01, f"{relative}: график пустой."

    report = pipeline.root / "results" / "report.txt"
    assert report.exists(), "main.py не создал results/report.txt."
    text = report.read_text(encoding="utf-8")
    assert "g" in text and "м/с" in text, (
        "results/report.txt не похож на отчёт: в нём нет результата g."
    )


def test_raw_journal_is_not_corrupted():
    raw = pd.read_csv(DATA_PATH)
    assert len(raw) == EXPECTED_ROWS, (
        "Файл data/measurements.csv изменён: ожидалось "
        f"{EXPECTED_ROWS} строк. Восстановите исходный журнал."
    )
    assert raw.isna().sum().sum() > 0, "В исходном журнале должны остаться пропуски."
    assert raw.duplicated().sum() > 0, "В исходном журнале должны остаться дубликаты."
