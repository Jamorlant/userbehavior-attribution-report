"""
从 outputs/ 的指标表生成报告图表, 输出到 figures/。

用法: python scripts/make_figures.py
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 170,
    "axes.edgecolor": "#B9BFC7",
    "axes.linewidth": 0.8,
})

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

C = {"pv": "#8FA7BF", "cart": "#E8A33D", "fav": "#6FA893", "buy": "#C0553F",
     "line": "#2F4B63", "grid": "#E6E9ED", "accent": "#C0553F", "muted": "#9AA4B0"}


def style(ax, title=None, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=C["grid"], lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12, color="#1B2733")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color="#4A5563")


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------- 图1: 用户级漏斗 ----------
def fig_funnel():
    f = pd.read_csv(OUT / "funnel_users.csv")
    steps = ["浏览", "意向(加购∪收藏)", "购买"]
    vals = [f.loc[f.step.str.startswith("浏览"), "users"].iloc[0],
            f.loc[f.step.str.startswith("意向"), "users"].iloc[0],
            f.loc[f.step.str.startswith("购买"), "users"].iloc[0]]

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    colors = [C["pv"], C["cart"], C["buy"]]
    bars = ax.barh(range(len(steps))[::-1], vals, color=colors, height=0.52, zorder=3)
    ax.set_yticks(range(len(steps))[::-1])
    ax.set_yticklabels(steps, fontsize=11)
    vmax = max(vals)
    drops = [None, 1 - vals[1] / vals[0], 1 - vals[2] / vals[1]]
    for b, v, dr in zip(bars, vals, drops):
        txt = f"{v:,} 人  ({v / vmax:.1%} of 浏览用户"
        txt += f" · 环比 −{dr:.1%})" if dr else ")"
        ax.text(v + vmax * 0.015, b.get_y() + b.get_height() / 2,
                txt, va="center", fontsize=10.5, color="#1B2733")
    ax.set_xlim(0, vmax * 1.52)
    ax.set_xlabel("去重用户数", fontsize=10, color="#4A5563")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=C["grid"], lw=0.9)
    ax.set_axisbelow(True)
    ax.set_xticks([])
    ax.set_title("图1  用户级漏斗: 每 100 个浏览用户只有 27.6 人下单",
                 loc="left", fontsize=13, fontweight="bold", pad=14, color="#1B2733")
    save(fig, "fig1_funnel_users.png")


# ---------- 图2: 每日行为趋势 ----------
def fig_daily():
    d = pd.read_csv(OUT / "by_date.csv")
    d["d"] = pd.to_datetime(d["d"])
    d = d[d.d > d.d.min()]          # 首日缺少前置购物车, 购买数系统性偏低, 不计入趋势判断

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.bar(d.d, d.pv / 1000, color=[C["pv"] if x.weekday() < 5 else "#C3CEDA" for x in d.d],
           width=0.62, label="浏览 pv(千)", zorder=3)
    ax.set_ylabel("浏览量(千)", fontsize=10, color="#4A5563")
    style(ax, None, "浏览量(千)")
    ax.set_title("图2  周末流量抬升, 但下单量没跟上", loc="left", fontsize=13, fontweight="bold",
                 pad=40, color="#1B2733")
    ax.set_ylim(0, d.pv.max() / 1000 * 1.45)

    ax2 = ax.twinx()
    ax2.plot(d.d, d.buy, color=C["buy"], lw=2.2, marker="o", ms=5, label="购买数", zorder=4)
    ax2.set_ylabel("购买数", fontsize=10, color="#4A5563")
    ax2.spines[["top", "left"]].set_visible(False)
    ax2.set_ylim(0, d.buy.max() * 1.55)
    for x, y in zip(d.d, d.buy):
        ax2.annotate(f"{int(y)}", (x, y), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=9, color=C["buy"])

    for x in d.d:
        if x.weekday() >= 5:
            ax.axvspan(x - pd.Timedelta(hours=12), x + pd.Timedelta(hours=12),
                       color="#F2F4F7", zorder=0)
    ax.tick_params(axis="x", labelsize=9)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(0, 1.16), ncol=2,
              frameon=False, fontsize=9.5)
    ax.text(0.995, -0.2, "灰底=周末;  11-25 为首日, 缺少前置购物车, 已剔除",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=C["muted"])
    save(fig, "fig2_daily_trend.png")


# ---------- 图3: 小时 × 意向/成交 ----------
def fig_hour():
    h = pd.read_csv(OUT / "by_hour.csv")
    ch = pd.read_csv(OUT / "cart_conv_by_hour.csv")
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.bar(h.hour, h.view_share * 100, color=C["pv"], width=0.72, label="浏览占比 %", zorder=3)
    style(ax, "图3  晚间 18-23 点: 47.5% 的流量, 加购率与成交率双高", "浏览占比 %")
    ax.set_xlabel("小时", fontsize=10, color="#4A5563")
    ax.set_xticks(range(0, 24, 2))
    ax.set_ylim(0, 12)

    ax2 = ax.twinx()
    ax2.plot(h.hour, h.intent_rate * 100, color=C["cart"], lw=2.2, marker="o", ms=4, label="加购率 %")
    ax2.plot(ch.cart_hour, ch.cart2buy * 100, color=C["buy"], lw=2.0, ls="--", marker="^", ms=4,
             label="加购→购买(按加购小时) %")
    ax2.spines[["top", "left"]].set_visible(False)
    ax2.set_ylabel("转化率 %", fontsize=10, color="#4A5563")
    ax2.set_ylim(0, 55)
    ax2.axvspan(17.6, 23.4, color="#FDF3EA", zorder=0)
    ax2.annotate("20 点: 加购后成交率 42.6%", xy=(20, 42.6), xytext=(13.2, 47),
                 fontsize=9.5, color=C["buy"], arrowprops=dict(arrowstyle="->", color=C["buy"], lw=0.9))
    ax2.annotate("4 点: 11.3%", xy=(4, 11.3), xytext=(5.4, 20), fontsize=9.5, color=C["muted"],
                 arrowprops=dict(arrowstyle="->", color=C["muted"], lw=0.9))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=9.5)
    save(fig, "fig3_hour_conversion.png")


# ---------- 图4: 星期 ----------
def fig_weekday():
    w = pd.read_csv(OUT / "by_weekday.csv")
    cw = pd.read_csv(OUT / "cart_conv_by_weekday.csv")
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    bars = ax.bar(w.label, w.pv / 1000, color=[C["pv"] if i < 5 else "#C3CEDA" for i in range(7)],
                  width=0.62, zorder=3, label="浏览量(千)")
    style(ax, "图4  周末的加购, 更难被兑现", "浏览量(千)")
    ax.set_ylim(0, w.pv.max() / 1000 * 1.28)

    ax2 = ax.twinx()
    ax2.plot(cw.label, cw.cart2buy * 100, color=C["buy"], lw=2.2, marker="o", ms=6,
             label="加购→购买率 %")
    ax2.set_ylim(0, 55)
    ax2.set_ylabel("加购→购买率 %", fontsize=10, color="#4A5563")
    ax2.spines[["top", "left"]].set_visible(False)
    for x, y in zip(cw.label, cw.cart2buy * 100):
        ax2.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=9.5, color=C["buy"])
    ax2.axhline(34.0, color=C["muted"], lw=1, ls=":", zorder=1, label="工作日均值 36.6%")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=9.5)
    save(fig, "fig4_weekday_conversion.png")


# ---------- 图5: 类目散点 ----------
def fig_category_scatter():
    c = pd.read_csv(OUT / "by_category.csv")
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    size = (c.pv / c.pv.max() * 1000) + 24
    ax.scatter(c.pv2cart * 100, c.cart2buy * 100, s=size, color=C["pv"], alpha=0.55,
               edgecolor="#4A5563", lw=0.6, zorder=3)

    mc, mb = c.pv2cart.mean() * 100, c.cart2buy.mean() * 100
    ax.axvline(mc, color=C["muted"], ls=":", lw=1.1, zorder=1)
    ax.axhline(mb, color=C["muted"], ls=":", lw=1.1, zorder=1)
    ax.text(mc + 0.08, 78, f"加购率均值 {mc:.1f}%", fontsize=8.5, color=C["muted"],
            rotation=90, va="top", ha="left")
    ax.text(1.87, mb - 3.2, f"加购→购买均值 {mb:.1f}%", fontsize=9, color=C["muted"])

    anom = c[(c.pv2cart > mc / 100) & (c.cart2buy < mb / 100)]
    ax.scatter(anom.pv2cart * 100, anom.cart2buy * 100, s=(anom.pv / c.pv.max() * 1000) + 24,
               color=C["buy"], alpha=0.75, edgecolor="#8C3226", lw=0.8, zorder=4,
               label="高加购 / 低成交 类目")
    ax.annotate(" / ".join(f"C-{int(v)}" for v in sorted(anom.category_id)),
                xy=(anom.pv2cart.min() * 100 - 0.35, anom.cart2buy.mean() * 100),
                xytext=(anom.pv2cart.min() * 100 - 1.15, anom.cart2buy.mean() * 100),
                fontsize=9, color="#8C3226", ha="right", va="center",
                arrowprops=dict(arrowstyle="->", color="#8C3226", lw=0.9))

    top = c.nlargest(3, "pv2buy")
    for _, r in top.iterrows():
        ax.annotate(f"C-{int(r.category_id)}", (r.pv2cart * 100, r.cart2buy * 100),
                    textcoords="offset points", xytext=(8, 6), fontsize=9, color="#2F4B63")

    style(ax, "图5  加购率高不等于能成交: 三个类目掉进右下方", "加购→购买率 %")
    ax.set_xlabel("浏览→加购率 %\n气泡大小 = 浏览量; 右下角为加购率高于均值、成交率不足均值一半的类目",
                  fontsize=10, color="#4A5563")
    ax.grid(axis="x", color=C["grid"], lw=0.9)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 80)
    ax.annotate("", xy=(mc + 1.0, mb - 9), xytext=(mc + 0.35, mb - 1.5),
                arrowprops=dict(arrowstyle="->", color=C["accent"], lw=1.2))
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    save(fig, "fig5_category_scatter.png")


# ---------- 图6: 类目转化排名 ----------
def fig_category_rank():
    c = pd.read_csv(OUT / "by_category.csv")
    c = c[c.pv >= 2000]
    top = c.nlargest(10, "pv2buy").assign(tag="转化最高 10")
    bot = c.nsmallest(10, "pv2buy").assign(tag="转化最低 10")
    d = pd.concat([bot, top]).sort_values("pv2buy")
    colors = [C["buy"] if t == "转化最高 10" else C["muted"] for t in d.tag]

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    ax.barh([f"C-{int(i)}" for i in d.category_id], d.pv2buy * 100, color=colors, height=0.62, zorder=3)
    style(ax, "图6  类目购买转化: 头部 3.3% vs 尾部 1.6%", "浏览→购买率 %")
    ax.axvline(c.pv2buy.mean() * 100, color=C["accent"], ls="--", lw=1.1, zorder=2)
    ax.annotate(f"全站均值 {c.pv2buy.mean()*100:.2f}%", xy=(c.pv2buy.mean() * 100 + 0.12, 0.4),
                fontsize=9, color=C["accent"])
    for y, (v, pv) in enumerate(zip(d.pv2buy * 100, d.pv)):
        ax.text(v + 0.08, y, f"{v:.1f}%  (浏览 {pv / 1000:.0f}k)", va="center", fontsize=8.5, color="#4A5563")
    ax.set_xlim(0, d.pv2buy.max() * 100 * 1.35)
    ax.tick_params(axis="y", labelsize=9)
    save(fig, "fig6_category_rank.png")


# ---------- 图7: 用户分层 ----------
def fig_segment():
    s = pd.read_csv(OUT / "by_user_segment.csv")
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    x = np.arange(len(s))
    w = 0.26
    ax.bar(x - w, s.user_share * 100, w, color=C["pv"], label="用户数占比 %", zorder=3)
    ax.bar(x, s.pv_share * 100, w, color=C["cart"], label="浏览量占比 %", zorder=3)
    ax.bar(x + w, s.buy_share * 100, w, color=C["buy"], label="订单占比 %", zorder=3)
    style(ax, "图7  31% 的高频用户撑起 64% 流量、57% 订单, 单位浏览转化却最低", "占比 %")
    ax.set_xticks(x)
    ax.set_xticklabels(s.segment)
    ax.set_ylim(0, 72)
    for i, r in s.iterrows():
        ax.text(i + w, r.buy_share * 100 + 1.6, f"{r.buy_share*100:.0f}%", ha="center", fontsize=8.5, color=C["buy"])
        ax.text(i - w, r.user_share * 100 + 1.6, f"{r.user_share*100:.0f}%", ha="center", fontsize=8.5, color=C["pv"])

    ax2 = ax.twinx()
    ax2.plot(x, s.pv2buy * 100, color="#1B2733", lw=2, marker="D", ms=6, label="浏览→购买率 %")
    for i, v in enumerate(s.pv2buy * 100):
        ax2.annotate(f"{v:.2f}%", (i, v), textcoords="offset points", xytext=(0, 10), ha="center",
                     fontsize=9.5, color="#1B2733")
    ax2.set_ylim(0, 4.4)
    ax2.set_ylabel("浏览→购买率 %", fontsize=10, color="#1B2733")
    ax2.spines[["top", "left"]].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", frameon=False, fontsize=9, ncol=4)
    save(fig, "fig7_user_segment.png")


# ---------- 图8: 加购→购买延迟 ----------
def fig_lag():
    lag = pd.read_csv(OUT / "cart_lag_hours.csv").lag_hours
    lag = lag[lag < 47.5]  # 48h 处的堆积是观测窗上限截断, 不属于真实分布
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    bins = np.arange(0, 49, 1)
    ax.hist(lag, bins=bins, color=C["cart"], alpha=0.85, zorder=3, label="已成交的购物车")
    style(ax, "图8  加购后的成交节奏: 中位 8.4 小时, 86% 在 24 小时内完成", "购物车数")
    ax.set_xlabel("加购 → 下单 间隔(小时)", fontsize=10, color="#4A5563")

    ax2 = ax.twinx()
    cdf = np.sort(lag.values)
    ax2.plot(cdf, np.arange(1, len(cdf) + 1) / len(cdf) * 100, color=C["line"], lw=2, label="累计占比 %")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("累计占比 %", fontsize=10, color="#4A5563")
    ax2.spines[["top", "left"]].set_visible(False)
    for h, label in ((8.4, "中位 8.4h"), (24, "24h 86%")):
        ax2.axvline(h, color=C["accent"], ls=":", lw=1.1, zorder=1)
        ax2.annotate(label, xy=(h, 8), xytext=(h - 0.4, 70), fontsize=9, color=C["accent"], rotation=90)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, fontsize=9.5)
    save(fig, "fig8_cart_lag.png")


def main():
    fig_funnel()
    fig_daily()
    fig_hour()
    fig_weekday()
    fig_category_scatter()
    fig_category_rank()
    fig_segment()
    fig_lag()
    print("\n".join(sorted(p.name for p in FIG.glob("*.png"))))
    print("figures:", len(list(FIG.glob("*.png"))))


if __name__ == "__main__":
    main()
