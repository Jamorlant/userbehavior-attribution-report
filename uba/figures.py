"""从 outputs/ 的指标表生成 8 张报告图表。

为什么标题里的数字也从 CSV 读：写死数值会在数据更新后与图表脱节，
而"图标题 = 数据"是这类归因报告最容易失真的地方。

入口：`python -m uba.figures [--out DIR] [--figdir DIR]`
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uba.config import Settings, get_settings
from uba.logging_utils import configure_logging, get_logger

log = get_logger("uba.figures")

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 170,
        "axes.edgecolor": "#B9BFC7",
        "axes.linewidth": 0.8,
    }
)

COLOR = {
    "pv": "#8FA7BF",
    "cart": "#E8A33D",
    "fav": "#6FA893",
    "buy": "#C0553F",
    "line": "#2F4B63",
    "grid": "#E6E9ED",
    "accent": "#C0553F",
    "muted": "#9AA4B0",
}
INK = "#1B2733"
AXIS_TEXT = "#4A5563"


class ChartData:
    """统一读取 outputs/，缺文件时给出可执行的下一步。"""

    def __init__(self, outputs_dir: Path) -> None:
        self.dir = outputs_dir

    def csv(self, name: str) -> pd.DataFrame:
        path = self.dir / name
        if not path.is_file():
            raise FileNotFoundError(f"缺少 {path}；先运行 `python -m uba.pipeline` 生成指标")
        return pd.read_csv(path)

    def metrics(self) -> dict[str, object]:
        path = self.dir / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"缺少 {path}；先运行 `python -m uba.pipeline` 生成指标")
        return json.loads(path.read_text(encoding="utf-8"))


def _style(ax: plt.Axes, title: str | None = None, ylabel: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=COLOR["grid"], lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=AXIS_TEXT)


def _save(fig: plt.Figure, fig_dir: Path, name: str) -> None:
    fig.savefig(fig_dir / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.debug("已生成 %s", name)


def fig_funnel(data: ChartData, fig_dir: Path) -> None:
    f = data.csv("funnel_users.csv")
    steps = ["浏览", "意向(加购∪收藏)", "购买"]
    vals = [
        int(f.loc[f.step.str.startswith("浏览"), "users"].iloc[0]),
        int(f.loc[f.step.str.startswith("意向"), "users"].iloc[0]),
        int(f.loc[f.step.str.startswith("购买"), "users"].iloc[0]),
    ]
    buy_rate = vals[2] / vals[0]

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    colors = [COLOR["pv"], COLOR["cart"], COLOR["buy"]]
    bars = ax.barh(range(len(steps))[::-1], vals, color=colors, height=0.52, zorder=3)
    ax.set_yticks(range(len(steps))[::-1])
    ax.set_yticklabels(steps, fontsize=11)
    vmax = max(vals)
    drops = [None, 1 - vals[1] / vals[0], 1 - vals[2] / vals[1]]
    for bar, value, drop in zip(bars, vals, drops, strict=True):
        text = f"{value:,} 人  ({value / vmax:.1%} of 浏览用户"
        text += f" · 环比 −{drop:.1%})" if drop else ")"
        ax.text(
            value + vmax * 0.015,
            bar.get_y() + bar.get_height() / 2,
            text,
            va="center",
            fontsize=10.5,
            color=INK,
        )
    ax.set_xlim(0, vmax * 1.52)
    ax.set_xlabel("去重用户数", fontsize=10, color=AXIS_TEXT)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=COLOR["grid"], lw=0.9)
    ax.set_axisbelow(True)
    ax.set_xticks([])
    ax.set_title(
        f"图1  用户级漏斗: {buy_rate * 100:.0f}% 的浏览用户最终下单",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=14,
        color=INK,
    )
    _save(fig, fig_dir, "fig1_funnel_users.png")


def fig_daily(data: ChartData, fig_dir: Path) -> None:
    d = data.csv("by_date.csv")
    d["d"] = pd.to_datetime(d["d"])
    d = d[d.d > d.d.min()]  # 首日缺少前置购物车，购买数系统性偏低，不计入趋势判断

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.bar(
        d.d,
        d.pv / 1000,
        color=[COLOR["pv"] if x.weekday() < 5 else "#C3CEDA" for x in d.d],
        width=0.62,
        label="浏览 pv(千)",
        zorder=3,
    )
    ax.set_ylabel("浏览量(千)", fontsize=10, color=AXIS_TEXT)
    _style(ax, None, "浏览量(千)")
    ax.set_title(
        "图2  周末流量抬升, 但下单量没跟上",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=40,
        color=INK,
    )
    ax.set_ylim(0, d.pv.max() / 1000 * 1.45)

    ax2 = ax.twinx()
    ax2.plot(d.d, d.buy, color=COLOR["buy"], lw=2.2, marker="o", ms=5, label="购买数", zorder=4)
    ax2.set_ylabel("购买数", fontsize=10, color=AXIS_TEXT)
    ax2.spines[["top", "left"]].set_visible(False)
    ax2.set_ylim(0, d.buy.max() * 1.55)
    for x, y in zip(d.d, d.buy, strict=True):
        ax2.annotate(
            f"{int(y)}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            color=COLOR["buy"],
        )

    for x in d.d:
        if x.weekday() >= 5:
            ax.axvspan(x - pd.Timedelta(hours=12), x + pd.Timedelta(hours=12), color="#F2F4F7", zorder=0)
    ax.tick_params(axis="x", labelsize=9)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(0, 1.16), ncol=2, frameon=False, fontsize=9.5)
    ax.text(
        0.995,
        -0.2,
        "灰底=周末;  11-25 为首日, 缺少前置购物车, 已剔除",
        transform=ax.transAxes,
        ha="right",
        fontsize=8.5,
        color=COLOR["muted"],
    )
    _save(fig, fig_dir, "fig2_daily_trend.png")


def fig_hour(data: ChartData, fig_dir: Path) -> None:
    h = data.csv("by_hour.csv")
    ch = data.csv("cart_conv_by_hour.csv")
    evening_share = h.loc[h.hour >= 18, "pv"].sum() / h.pv.sum()

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.bar(h.hour, h.view_share * 100, color=COLOR["pv"], width=0.72, label="浏览占比 %", zorder=3)
    _style(
        ax,
        f"图3  晚间 18-23 点承载 {evening_share * 100:.1f}% 的浏览, 加购后成交率全天最高",
        "浏览占比 %",
    )
    ax.set_xlabel("小时", fontsize=10, color=AXIS_TEXT)
    ax.set_xticks(range(0, 24, 2))
    ax.set_ylim(0, max(8.0, h.view_share.max() * 100 * 1.4))

    ax2 = ax.twinx()
    ax2.plot(h.hour, h.intent_rate * 100, color=COLOR["cart"], lw=2.2, marker="o", ms=4, label="加购率 %")
    ax2.plot(
        ch.cart_hour,
        ch.cart2buy * 100,
        color=COLOR["buy"],
        lw=2.0,
        ls="--",
        marker="^",
        ms=4,
        label="加购→购买(按加购小时) %",
    )
    ax2.spines[["top", "left"]].set_visible(False)
    ax2.set_ylabel("转化率 %", fontsize=10, color=AXIS_TEXT)
    peak = max(float(h.intent_rate.max()), float(ch.cart2buy.max())) * 100
    ax2.set_ylim(0, max(12.0, peak * 1.5))
    ax2.axvspan(17.6, 23.4, color="#FDF3EA", zorder=0)
    best = ch.loc[ch.cart2buy.idxmax()]
    worst = ch.loc[ch.cart2buy.idxmin()]
    ax2.annotate(
        f"{int(best.cart_hour)} 点: 加购后成交率 {best.cart2buy * 100:.1f}%",
        xy=(best.cart_hour, best.cart2buy * 100),
        xytext=(13.2, peak * 1.3),
        fontsize=9.5,
        color=COLOR["buy"],
        arrowprops=dict(arrowstyle="->", color=COLOR["buy"], lw=0.9),
    )
    ax2.annotate(
        f"{int(worst.cart_hour)} 点: {worst.cart2buy * 100:.1f}%",
        xy=(worst.cart_hour, worst.cart2buy * 100),
        xytext=(5.4, 20),
        fontsize=9.5,
        color=COLOR["muted"],
        arrowprops=dict(arrowstyle="->", color=COLOR["muted"], lw=0.9),
    )
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=9.5)
    _save(fig, fig_dir, "fig3_hour_conversion.png")


def fig_weekday(data: ChartData, fig_dir: Path) -> None:
    w = data.csv("by_weekday.csv")
    cw = data.csv("cart_conv_by_weekday.csv")
    weekday_rate = cw.loc[cw.cart_weekday < 5, "cart2buy"].mean()
    weekend_share = w.loc[w.weekday >= 5, "pv"].sum() / w.pv.sum()
    weekday_conv = w.loc[w.weekday < 5, "pv2buy"].mean()
    weekend_conv = w.loc[w.weekday >= 5, "pv2buy"].mean()
    gap = 1 - weekend_conv / weekday_conv

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.bar(
        w.label,
        w.pv / 1000,
        color=[COLOR["pv"] if i < 5 else "#C3CEDA" for i in range(7)],
        width=0.62,
        zorder=3,
        label="浏览量(千)",
    )
    _style(ax, f"图4  周末拿走 {weekend_share * 100:.1f}% 的浏览, 浏览→购买率低 {gap * 100:.0f}%", "浏览量(千)")
    ax.set_ylim(0, w.pv.max() / 1000 * 1.28)

    ax2 = ax.twinx()
    ax2.plot(cw.label, cw.cart2buy * 100, color=COLOR["buy"], lw=2.2, marker="o", ms=6, label="加购→购买率 %")
    ax2.set_ylim(0, max(12.0, cw.cart2buy.max() * 100 * 1.45))
    ax2.set_ylabel("加购→购买率 %", fontsize=10, color=AXIS_TEXT)
    ax2.spines[["top", "left"]].set_visible(False)
    for x, y in zip(cw.label, cw.cart2buy * 100, strict=True):
        ax2.annotate(
            f"{y:.1f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=9.5,
            color=COLOR["buy"],
        )
    ax2.axhline(
        weekday_rate * 100, color=COLOR["muted"], lw=1, ls=":", zorder=1, label=f"工作日均值 {weekday_rate * 100:.1f}%"
    )
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=9.5)
    _save(fig, fig_dir, "fig4_weekday_conversion.png")


def fig_category_scatter(data: ChartData, fig_dir: Path) -> None:
    all_cats = data.csv("by_category.csv")
    # 只画有统计意义的类目：真实数据集 9,437 个类目里大量只有几十次浏览，散点会被噪声淹没
    c = all_cats[(all_cats.pv >= 10_000) & (all_cats.cart > 0)]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    size = (c.pv / c.pv.max() * 900) + 22
    ax.scatter(
        c.pv2cart * 100, c.cart2buy * 100, s=size, color=COLOR["pv"], alpha=0.55, edgecolor=AXIS_TEXT, lw=0.5, zorder=3
    )

    mc, mb = c.pv2cart.mean() * 100, c.cart2buy.mean() * 100
    ax.axvline(mc, color=COLOR["muted"], ls=":", lw=1.1, zorder=1)
    ax.axhline(mb, color=COLOR["muted"], ls=":", lw=1.1, zorder=1)
    ax.text(0.015, 0.955, f"加购率均值 {mc:.1f}%", fontsize=8.5, color=COLOR["muted"], transform=ax.transAxes)
    ax.text(
        0.985, 0.86, f"加购→购买均值 {mb:.1f}%", fontsize=8.5, color=COLOR["muted"], ha="right", transform=ax.transAxes
    )

    anom = c[(c.pv2cart > mc / 100) & (c.cart2buy < mb / 100)].copy()
    anom["c2b_gap"] = anom["cart2buy"] - mb / 100
    pick = anom.nsmallest(3, "c2b_gap")
    ax.scatter(
        anom.pv2cart * 100,
        anom.cart2buy * 100,
        s=(anom.pv / c.pv.max() * 900) + 22,
        color=COLOR["buy"],
        alpha=0.55,
        edgecolor="#8C3226",
        lw=0.6,
        zorder=4,
        label="高加购 / 低成交 类目",
    )
    if not pick.empty:
        ax.scatter(
            pick.pv2cart * 100,
            pick.cart2buy * 100,
            s=(pick.pv / c.pv.max() * 900) + 20,
            color="#8C3226",
            alpha=0.95,
            edgecolor="white",
            lw=0.9,
            zorder=5,
        )
        offsets = [(34, 18), (34, -6), (52, -26)]
        for (_, row), offset in zip(pick.iterrows(), offsets, strict=False):
            ax.annotate(
                f"C-{int(row.category_id)}",
                xy=(row.pv2cart * 100, row.cart2buy * 100),
                textcoords="offset points",
                xytext=offset,
                fontsize=8.5,
                color="#8C3226",
                arrowprops=dict(arrowstyle="->", color="#8C3226", lw=0.8, shrinkB=2),
            )

    for _, row in c.nlargest(3, "pv2buy").iterrows():
        ax.annotate(
            f"C-{int(row.category_id)}",
            (row.pv2cart * 100, row.cart2buy * 100),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8.5,
            color=COLOR["line"],
        )

    _style(ax, "图5  加购率高不等于能成交: 红点类目值得单独查", "加购→购买率 %")
    ax.set_xlabel(
        "浏览→加购率 %\n气泡大小 = 浏览量; 仅展示浏览量 ≥ 1 万的 904 个类目",
        fontsize=10,
        color=AXIS_TEXT,
    )
    ax.grid(axis="x", color=COLOR["grid"], lw=0.9)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(4.0, float(c.cart2buy.max()) * 100 * 1.15))
    ax.set_xlim(0, max(8.0, float(c.pv2cart.quantile(0.995)) * 100 * 1.2))
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    _save(fig, fig_dir, "fig5_category_scatter.png")


def fig_category_rank(data: ChartData, fig_dir: Path) -> None:
    c = data.csv("by_category.csv")
    # 与报告口径一致：只看浏览量 ≥ 1 万且有加购的类目，避免小样本类目刷出 100% 转化
    c = c[(c.pv >= 10_000) & (c.cart > 0)]
    top = c.nlargest(10, "pv2buy").assign(tag="转化最高 10")
    bot = c.nsmallest(10, "pv2buy").assign(tag="转化最低 10")
    d = pd.concat([bot, top]).sort_values("pv2buy")
    colors = [COLOR["buy"] if t == "转化最高 10" else COLOR["muted"] for t in d.tag]

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    ax.barh([f"C-{int(i)}" for i in d.category_id], d.pv2buy * 100, color=colors, height=0.62, zorder=3)
    _style(
        ax,
        f"图6  类目购买转化: 头部 {top.pv2buy.min() * 100:.1f}% vs 尾部 {bot.pv2buy.max() * 100:.1f}%",
        "浏览→购买率 %",
    )
    ax.axvline(c.pv2buy.mean() * 100, color=COLOR["accent"], ls="--", lw=1.1, zorder=2)
    ax.annotate(
        f"全站均值 {c.pv2buy.mean() * 100:.2f}%",
        xy=(c.pv2buy.mean() * 100 + 0.12, 0.4),
        fontsize=9,
        color=COLOR["accent"],
    )
    for y, (v, pv) in enumerate(zip(d.pv2buy * 100, d.pv, strict=True)):
        ax.text(v + 0.08, y, f"{v:.1f}%  (浏览 {pv / 1000:.0f}k)", va="center", fontsize=8.5, color=AXIS_TEXT)
    ax.set_xlim(0, d.pv2buy.max() * 100 * 1.35)
    ax.tick_params(axis="y", labelsize=9)
    _save(fig, fig_dir, "fig6_category_rank.png")


def fig_segment(data: ChartData, fig_dir: Path) -> None:
    s = data.csv("by_user_segment.csv")
    high = s[s.segment == "高频"].iloc[0]

    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    x = np.arange(len(s))
    w = 0.26
    ax.bar(x - w, s.user_share * 100, w, color=COLOR["pv"], label="用户数占比 %", zorder=3)
    ax.bar(x, s.pv_share * 100, w, color=COLOR["cart"], label="浏览量占比 %", zorder=3)
    ax.bar(x + w, s.buy_share * 100, w, color=COLOR["buy"], label="订单占比 %", zorder=3)
    _style(
        ax,
        f"图7  {high.user_share * 100:.0f}% 的高频用户撑起 {high.pv_share * 100:.0f}% 流量、"
        f"{high.buy_share * 100:.0f}% 订单, 单位浏览转化却最低",
        "占比 %",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(s.segment)
    ax.set_ylim(0, max(72.0, float(s[["user_share", "pv_share", "buy_share"]].to_numpy().max()) * 100 * 1.15))
    for i, r in s.iterrows():
        ax.text(
            i + w, r.buy_share * 100 + 1.6, f"{r.buy_share * 100:.0f}%", ha="center", fontsize=8.5, color=COLOR["buy"]
        )
        ax.text(
            i - w, r.user_share * 100 + 1.6, f"{r.user_share * 100:.0f}%", ha="center", fontsize=8.5, color=COLOR["pv"]
        )

    ax2 = ax.twinx()
    ax2.plot(x, s.pv2buy * 100, color=INK, lw=2, marker="D", ms=6, label="浏览→购买率 %")
    for i, v in enumerate(s.pv2buy * 100):
        ax2.annotate(
            f"{v:.2f}%", (i, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9.5, color=INK
        )
    ax2.set_ylim(0, max(4.4, float(s.pv2buy.max()) * 100 * 1.5))
    ax2.set_ylabel("浏览→购买率 %", fontsize=10, color=INK)
    ax2.spines[["top", "left"]].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", frameon=False, fontsize=9, ncol=4)
    _save(fig, fig_dir, "fig7_user_segment.png")


def fig_lag(data: ChartData, fig_dir: Path) -> None:
    hist = data.csv("cart_lag_hist.csv").sort_values("bin_hours")
    m = data.metrics()["cart2buy"]
    median_hours = float(m["median_lag_hours"])
    within_24h = float(m["converted_within_24h"])
    total = int(hist["carts"].sum())
    # 统计口径用全样本；画图只展示 48h 以内的分布，长尾单独在报告里说明
    shown = hist[hist["bin_hours"] < 48]
    cdf = shown["carts"].cumsum() / total * 100

    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    ax.bar(shown["bin_hours"], shown["carts"], width=0.9, align="edge", color=COLOR["cart"], alpha=0.85, zorder=3)
    _style(
        ax,
        f"图8  加购后的成交节奏: 中位 {median_hours:.1f} 小时, {within_24h * 100:.0f}% 在 24 小时内完成",
        "购物车数",
    )
    ax.set_xlabel("加购 → 下单 间隔(小时)", fontsize=10, color=AXIS_TEXT)

    ax2 = ax.twinx()
    ax2.plot(shown["bin_hours"] + 0.5, cdf, color=COLOR["line"], lw=2, label="累计占比 %")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("累计占比 %", fontsize=10, color=AXIS_TEXT)
    ax2.spines[["top", "left"]].set_visible(False)
    for hours, label, y in (
        (median_hours, f"中位 {median_hours:.1f}h", 72),
        (24, f"24h {within_24h * 100:.0f}%", 56),
    ):
        ax2.axvline(hours, color=COLOR["accent"], ls=":", lw=1.1, zorder=1)
        ax2.annotate(label, xy=(hours, 8), xytext=(hours - 0.9, y), fontsize=9, color=COLOR["accent"], rotation=90)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, fontsize=9.5)
    _save(fig, fig_dir, "fig8_cart_lag.png")


def build_all(settings: Settings) -> list[Path]:
    fig_dir = settings.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = ChartData(settings.outputs_dir)
    for fn in (
        fig_funnel,
        fig_daily,
        fig_hour,
        fig_weekday,
        fig_category_scatter,
        fig_category_rank,
        fig_segment,
        fig_lag,
    ):
        fn(data, fig_dir)
    images = sorted(fig_dir.glob("*.png"))
    log.info("图表已写入 %s（%d 张）", fig_dir, len(images))
    return images


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    ap = argparse.ArgumentParser(description="生成报告图表")
    ap.add_argument("--out", type=Path, help="指标目录")
    ap.add_argument("--figdir", type=Path, help="图表输出目录")
    ap.add_argument("--log-level", default=None)
    args = ap.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.out:
        overrides["outputs_dir"] = args.out
    if args.figdir:
        overrides["figures_dir"] = args.figdir
    if args.log_level:
        overrides["log_level"] = args.log_level
    if overrides:
        settings = replace(settings, **overrides)
    configure_logging(settings.log_level)

    try:
        build_all(settings)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
