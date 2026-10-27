"""Полная обработка данных лабораторной работы № 1.

Скрипт вызывает функции из lab.py и сохраняет результаты и графики:

    results/cleaning.txt
    results/summary.csv
    results/g_result.json
    results/report.txt
    plots/a_vs_sin_alpha.png
    plots/g_vs_angle.png

Запуск:
    python main.py

Этот файл можно менять, но структура перечисленных файлов должна сохраниться.
"""

from __future__ import annotations

import json
from pathlib import Path

import lab

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    # 1. Журнал измерений и проверка качества
    raw = lab.load_data(lab.DATA_PATH)
    n_raw = len(raw)
    missing = raw.isna().sum()
    n_duplicates = int(raw.duplicated().sum())

    cleaning_log = (
        "Проверка качества данных\n"
        "========================\n"
        f"Строк в журнале: {n_raw}\n"
        f"Пропуски по колонкам:\n{missing.to_string()}\n"
        f"Полных дубликатов строк: {n_duplicates}\n"
    )

    # 2. Очистка
    clean = lab.clean_data(raw)
    cleaning_log += (
        f"Строк после очистки: {len(clean)}\n"
        f"Удалено строк: {n_raw - len(clean)}\n"
    )
    (RESULTS_DIR / "cleaning.txt").write_text(cleaning_log, encoding="utf-8")
    print(cleaning_log)

    # 3. Ускорения, сводка по углам и МНК
    df = clean.copy()
    df["acceleration_ms2"] = lab.acceleration(df["distance_m"], df["time_s"])
    summary = lab.summarize_by_angle(df)
    g, g_err, b = lab.fit_g(summary)

    # 4. Таблица и машинно-читаемый результат
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False, float_format="%.5f")
    payload = {
        "g": round(float(g), 6),
        "g_err": round(float(g_err), 6),
        "intercept": round(float(b), 6),
    }
    (RESULTS_DIR / "g_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # 5. Графики
    lab.plot_a_vs_sin(summary, PLOTS_DIR / "a_vs_sin_alpha.png")
    lab.plot_g_vs_angle(summary, PLOTS_DIR / "g_vs_angle.png")

    # 6. Текстовый отчёт
    deviation = abs(g - lab.G_TABLE)
    report = (
        "Лабораторная работа № 1. Наклонная плоскость\n"
        "=============================================\n\n"
        f"Строк в журнале: {n_raw}, после очистки: {len(clean)}\n"
        f"Удалено строк: {n_raw - len(clean)}\n"
        f"Число углов: {len(summary)}\n\n"
        "Сводная таблица по углам:\n"
        f"{summary.round(4).to_string(index=False)}\n\n"
        "Результат МНК (a = g*sin(alpha) + b):\n"
        f"  g = {g:.3f} +- {g_err:.3f} м/с^2\n"
        f"  b = {b:+.3f} м/с^2\n\n"
        f"Табличное значение: g = {lab.G_TABLE:.2f} м/с^2\n"
        f"Расхождение: {deviation:.3f} м/с^2"
        f" ({deviation / g_err:.1f} sigma_g)\n"
    )
    (RESULTS_DIR / "report.txt").write_text(report, encoding="utf-8")

    print(report)
    print("Готово: results/ и plots/ обновлены.")


if __name__ == "__main__":
    main()
