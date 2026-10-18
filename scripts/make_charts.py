"""Графики для слайдов о формуле нейросети.

Генерирует в web/static/images/:
  chart-linear.png  — стоимость = количество × цена (одна переменная);
  chart-plane.png   — стоимость = 30·капуста + 50·картошка (две переменные);
  chart-contour.png — линии равной стоимости (запасной вариант).

Покупка для отметки: 2 кг капусты и 4 кг картошки (итого 260 ₽).

Запуск:
    .venv/bin/python scripts/make_charts.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "web" / "static" / "images"

POTATO_PRICE = 50
CABBAGE_PRICE = 30

POTATO_KG = 4
CABBAGE_KG = 2
POTATO_COST = POTATO_KG * POTATO_PRICE
CABBAGE_COST = CABBAGE_KG * CABBAGE_PRICE
TOTAL_COST = POTATO_COST + CABBAGE_COST


def make_linear():
    x = np.linspace(0, 10, 200)

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    ax.plot(x, POTATO_PRICE * x, color="#667eea", lw=4, label=f"картошка: {POTATO_PRICE} ₽/кг")
    ax.plot(x, CABBAGE_PRICE * x, color="#764ba2", lw=4, label=f"капуста: {CABBAGE_PRICE} ₽/кг")

    ax.scatter([POTATO_KG], [POTATO_COST], color="#667eea", s=170, zorder=5, edgecolor="#333")
    ax.scatter([CABBAGE_KG], [CABBAGE_COST], color="#764ba2", s=170, zorder=5, edgecolor="#333")

    summary = (
        f"{POTATO_KG} кг картошки = {POTATO_COST} ₽\n"
        f"{CABBAGE_KG} кг капусты = {CABBAGE_COST} ₽\n"
        f"Итого = {TOTAL_COST} ₽"
    )
    ax.text(
        6.3, 20, summary,
        fontsize=13, linespacing=1.6, color="#333",
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#dfe3f0"),
    )

    ax.set_xlabel("Количество, кг", fontsize=18)
    ax.set_ylabel("Стоимость, ₽", fontsize=18)
    ax.tick_params(labelsize=15)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 520)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=15, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "chart-linear.png", facecolor="white")
    plt.close(fig)


def make_plane():
    cabbage = np.linspace(0, 10, 25)
    potato = np.linspace(0, 10, 25)
    cabbage_grid, potato_grid = np.meshgrid(cabbage, potato)
    cost_grid = CABBAGE_PRICE * cabbage_grid + POTATO_PRICE * potato_grid

    fig = plt.figure(figsize=(8, 4.8), dpi=200)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.plot_surface(
        cabbage_grid, potato_grid, cost_grid,
        cmap="viridis", alpha=0.45, edgecolor="none", zorder=1,
    )

    # три пунктирные линии проекции:
    # (0,4,0)-(2,4,0), (2,0,0)-(2,4,0), (2,4,0)-(2,4,260)
    ax.plot(
        [0, CABBAGE_KG], [POTATO_KG, POTATO_KG], [0, 0],
        color="#d32f2f", linestyle="--", linewidth=2.5, zorder=10,
    )
    ax.plot(
        [CABBAGE_KG, CABBAGE_KG], [0, POTATO_KG], [0, 0],
        color="#d32f2f", linestyle="--", linewidth=2.5, zorder=10,
    )
    ax.plot(
        [CABBAGE_KG, CABBAGE_KG], [POTATO_KG, POTATO_KG], [0, TOTAL_COST],
        color="#d32f2f", linestyle="--", linewidth=2.5, zorder=10,
    )

    ax.scatter(
        [CABBAGE_KG], [POTATO_KG], [TOTAL_COST],
        color="#ffc107", s=220, edgecolor="#333", linewidth=1.5,
        depthshade=False, zorder=12,
    )
    ax.scatter(
        [CABBAGE_KG], [POTATO_KG], [0],
        color="#ffc107", s=110, edgecolor="#333", linewidth=1.5,
        depthshade=False, zorder=11,
    )
    ax.text(
        CABBAGE_KG, POTATO_KG, TOTAL_COST + 60,
        f"{TOTAL_COST} ₽",
        fontsize=14, fontweight="bold", color="#333", zorder=13,
    )

    ax.set_xlabel("капуста, кг", fontsize=15, labelpad=10)
    ax.set_ylabel("картошка, кг", fontsize=15, labelpad=10)
    ax.set_zlabel("стоимость, ₽", fontsize=15, labelpad=8)
    ax.tick_params(labelsize=11)
    ax.set_zlim(0, 850)
    ax.view_init(elev=24, azim=-58)

    fig.tight_layout()
    fig.savefig(OUT / "chart-plane.png", facecolor="white")
    plt.close(fig)


def make_contour():
    cabbage = np.linspace(0, 10, 200)
    potato = np.linspace(0, 10, 200)
    cabbage_grid, potato_grid = np.meshgrid(cabbage, potato)
    cost_grid = CABBAGE_PRICE * cabbage_grid + POTATO_PRICE * potato_grid

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    levels = [50, 100, 150, 200, 300, 400]
    contours = ax.contour(
        cabbage_grid, potato_grid, cost_grid,
        levels=levels, colors="#667eea", linewidths=2.5,
    )
    ax.clabel(contours, inline=True, fontsize=13, fmt="%d ₽")
    ax.scatter([CABBAGE_KG], [POTATO_KG], color="#ffc107", s=170, zorder=5, edgecolor="#333")

    ax.set_xlabel("капуста, кг", fontsize=18)
    ax.set_ylabel("картошка, кг", fontsize=18)
    ax.tick_params(labelsize=15)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "chart-contour.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_linear()
    make_plane()
    make_contour()
    print("Готово:", OUT / "chart-linear.png", OUT / "chart-plane.png", OUT / "chart-contour.png")
