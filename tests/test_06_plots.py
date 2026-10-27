"""Проверки графиков (задание 4.6)."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib import image as mpimg
from matplotlib import pyplot as plt

import lab


def _check_png(path, title):
    assert path.exists(), (
        f"{title}: файл не создан. Проверьте, что функция сохраняет рисунок "
        "именно по переданному пути out_path."
    )
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", (
        f"{title}: файл не является PNG. Сохраняйте с расширением .png "
        "(savefig сам определит формат)."
    )
    assert len(data) > 8_000, (
        f"{title}: файл подозрительно маленький ({len(data)} байт). "
        "Возможно, сохранилась пустая фигура."
    )

    image = mpimg.imread(path)
    assert image.ndim in (2, 3), f"{title}: не удалось прочитать изображение."
    height, width = image.shape[:2]
    assert width >= 400 and height >= 300, (
        f"{title}: размер {width}×{height} px слишком мал. "
        "Сделайте figsize не меньше 6×4 дюймов и dpi >= 150."
    )
    assert float(np.std(image)) > 0.01, (
        f"{title}: изображение почти однотонное — похоже, на нём нет данных. "
        "Проверьте, что вы рисуете точки/линии и сохраняете нужную фигуру."
    )


def test_plot_a_vs_sin(tmp_path, ref):
    out = tmp_path / "a_vs_sin.png"
    result = lab.plot_a_vs_sin(ref.summary, out)
    plt.close("all")

    _check_png(out, "График a от sin(alpha)")
    assert result is None or str(result) == str(out), (
        "plot_a_vs_sin может ничего не возвращать, но если возвращает путь, "
        "это должен быть путь сохранённого файла."
    )


def test_plot_g_vs_angle(tmp_path, ref):
    out = tmp_path / "g_vs_angle.png"
    result = lab.plot_g_vs_angle(ref.summary, out)
    plt.close("all")

    _check_png(out, "График g от угла")
    assert result is None or str(result) == str(out), (
        "plot_g_vs_angle может ничего не возвращать, но если возвращает путь, "
        "это должен быть путь сохранённого файла."
    )


def test_plots_are_different(tmp_path, ref):
    first = tmp_path / "a.png"
    second = tmp_path / "g.png"
    lab.plot_a_vs_sin(ref.summary, first)
    lab.plot_g_vs_angle(ref.summary, second)
    plt.close("all")

    assert first.read_bytes() != second.read_bytes(), (
        "Оба графика получились одинаковыми. Это должны быть разные зависимости: "
        "a(sin alpha) и g(alpha)."
    )
