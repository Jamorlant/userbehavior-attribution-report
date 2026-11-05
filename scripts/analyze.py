"""
数据清洗 + 漏斗 + 归因指标计算。

输入: data/userbehavior.csv.gz  (schema: user_id,item_id,category_id,behavior_type,timestamp)
输出: outputs/*.csv, outputs/metrics.json

口径说明:
  - 行为级转化率 = 目标行为数 / 浏览数(pv)
  - 加购→购买(序列口径): 同一 (user_id, item_id) 上, 加购之后是否出现购买事件
  - 用户级漏斗 = 去重用户数递进, 用于判断"有多少人走到了下一步"
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "userbehavior.csv.gz"
OUT = ROOT / "outputs"

VALID_TYPES = ["pv", "cart", "fav", "buy"]
# 一周按中文习惯从周一开始
WD_LABELS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def load_and_clean():
    df = pd.read_csv(DATA)
    raw_rows = len(df)

    df = df.dropna(subset=["user_id", "item_id", "category_id", "behavior_type", "timestamp"])
    df = df[df.behavior_type.isin(VALID_TYPES)]
    df = df.drop_duplicates()
    df["timestamp"] = df.timestamp.astype("int64")
    t0, t1 = df.timestamp.min(), df.timestamp.max()
    df = df[(df.timestamp >= t0) & (df.timestamp < t1 + 1)]
    df["dt"] = pd.to_datetime(df.timestamp, unit="s")
    df["hour"] = df.dt.dt.hour
    df["weekday"] = df.dt.dt.weekday

    audit = {
        "raw_rows": int(raw_rows),
        "clean_rows": int(len(df)),
        "dropped_rows": int(raw_rows - len(df)),
        "dup_rows_removed": int(raw_rows - len(df.drop_duplicates())),
        "users": int(df.user_id.nunique()),
        "items": int(df.item_id.nunique()),
        "categories": int(df.category_id.nunique()),
        "date_start": str(df.dt.min()),
        "date_end": str(df.dt.max()),
        "days": int(df.dt.dt.date.nunique()),
        "behavior_counts": {k: int(v) for k, v in df.behavior_type.value_counts().items()},
    }
    return df, audit


def wilson_ci(k, n, z=1.96):
    """二项比例的 Wilson 置信区间, 避免小样本下的正态近似失真。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return float((c - r) / d), float((c + r) / d)


def funnel(df):
    """用户级漏斗: 浏览 → 意向(加购或收藏) → 购买"""
    def users_of(typeset):
        return set(df.loc[df.behavior_type.isin(typeset), "user_id"].unique())

    u_pv = users_of({"pv"})
    u_intent = users_of({"cart", "fav"})
    u_cart = users_of({"cart"})
    u_fav = users_of({"fav"})
    u_buy = users_of({"buy"})

    rows = [
        ("浏览 (pv)", len(u_pv), None),
        ("加购 (cart)", len(u_cart), len(u_pv)),
        ("收藏 (fav)", len(u_fav), len(u_pv)),
        ("意向 (加购∪收藏)", len(u_intent), len(u_pv)),
        ("购买 (buy)", len(u_buy), len(u_intent)),
    ]
    out = []
    for name, n, base in rows:
        rate = (n / base) if base else np.nan
        lo, hi = wilson_ci(n, base) if base else (np.nan, np.nan)
        out.append(dict(step=name, users=n, base_users=base, rate=rate, ci_low=lo, ci_high=hi))
    f = pd.DataFrame(out)

    # 行为级加购/收藏率
    cnt = df.behavior_type.value_counts()
    pv = cnt["pv"]
    behavior = {
        "pv": int(pv),
        "cart": int(cnt["cart"]),
        "fav": int(cnt["fav"]),
        "buy": int(cnt["buy"]),
        "pv2cart": float(cnt["cart"] / pv),
        "pv2fav": float(cnt["fav"] / pv),
        "pv2intent": float((cnt["cart"] + cnt["fav"]) / pv),
        "pv2buy": float(cnt["buy"] / pv),
        "intent2buy_users": float(len(u_buy) / len(u_intent)),
        "pv2buy_users": float(len(u_buy) / len(u_pv)),
    }
    return f, behavior


def cart_to_buy(df):
    """序列口径: 同一 (user, item) 加购之后是否下单, 附带 24h 窗口。"""
    carts = (df[df.behavior_type == "cart"]
             .groupby(["user_id", "item_id"], as_index=False)
             .agg(cart_ts=("timestamp", "min"),
                  category_id=("category_id", "first")))
    buys = (df[df.behavior_type == "buy"]
            .groupby(["user_id", "item_id"], as_index=False)
            .agg(buy_ts=("timestamp", "min")))

    m = carts.merge(buys, on=["user_id", "item_id"], how="left")
    m["converted"] = (m.buy_ts.notna()) & (m.buy_ts >= m.cart_ts)
    m["converted_24h"] = m.converted & ((m.buy_ts - m.cart_ts) <= 86400)

    # 观测窗口校正: 只统计"还有足够观察时间"的购物车(至少 48h), 否则末尾的购物车会被系统性判为未转化
    t_max = int(df.timestamp.max())
    W = 48 * 3600
    censored = int((m.cart_ts > t_max - W).sum())
    m = m[m.cart_ts <= t_max - W].copy()

    by_cat = (m.groupby("category_id")
              .agg(carted_pairs=("converted", "size"),
                   converted=("converted", "sum"),
                   converted_24h=("converted_24h", "sum"))
              .reset_index())
    by_cat["cart2buy"] = by_cat.converted / by_cat.carted_pairs
    by_cat["cart2buy_24h"] = by_cat.converted_24h / by_cat.carted_pairs

    overall = dict(
        carted_pairs=int(len(m)),
        converted_pairs=int(m.converted.sum()),
        cart2buy=float(m.converted.mean()),
        cart2buy_24h=float(m.converted_24h.mean()),
        median_lag_hours=float(np.median((m.loc[m.converted, "buy_ts"] - m.loc[m.converted, "cart_ts"]) / 3600)),
        censored_pairs=censored,
    )
    lag = ((m.loc[m.converted, "buy_ts"] - m.loc[m.converted, "cart_ts"]) / 3600)
    pd.DataFrame({"lag_hours": lag}).to_csv(OUT / "cart_lag_hours.csv", index=False)

    # 按"加购发生时段"归因转化: 避免下单时间滞后污染时段结论
    m["cart_hour"] = pd.to_datetime(m.cart_ts, unit="s").dt.hour
    m["cart_weekday"] = pd.to_datetime(m.cart_ts, unit="s").dt.weekday
    conv_hour = (m.groupby("cart_hour")
                 .agg(carts=("converted", "size"), converted=("converted", "sum"))
                 .reindex(range(24), fill_value=0).reset_index())
    conv_hour["cart2buy"] = conv_hour.converted / conv_hour.carts.replace(0, np.nan)
    conv_hour["cart_share"] = conv_hour.carts / conv_hour.carts.sum()
    conv_hour.to_csv(OUT / "cart_conv_by_hour.csv", index=False, encoding="utf-8-sig")

    conv_wd = (m.groupby("cart_weekday")
               .agg(carts=("converted", "size"), converted=("converted", "sum"))
               .reindex(range(7), fill_value=0).reset_index())
    conv_wd["label"] = conv_wd.cart_weekday.map(dict(enumerate(WD_LABELS)))
    conv_wd["cart2buy"] = conv_wd.converted / conv_wd.carts.replace(0, np.nan)
    conv_wd.to_csv(OUT / "cart_conv_by_weekday.csv", index=False, encoding="utf-8-sig")
    overall["converted_within_1h"] = float((lag <= 1).mean())
    overall["converted_within_24h"] = float((lag <= 24).mean())
    # 未经过加购的直接购买占比
    buys_total = int((df.behavior_type == "buy").sum())
    buys_with_cart = int(df.loc[df.behavior_type == "buy", ["user_id", "item_id"]]
                         .merge(carts[["user_id", "item_id"]], on=["user_id", "item_id"], how="inner")
                         .shape[0])
    overall["buy_events"] = buys_total
    overall["buy_share_without_cart"] = float(1 - buys_with_cart / buys_total)
    return by_cat, overall, conv_hour, conv_wd


def by_category(df, cat_seq):
    g = df.groupby("category_id").behavior_type.value_counts().unstack(fill_value=0).reset_index()
    for col in VALID_TYPES:
        if col not in g:
            g[col] = 0
    g["pv2cart"] = g.cart / g.pv
    g["pv2fav"] = g.fav / g.pv
    g["pv2buy"] = g.buy / g.pv
    g["intent_rate"] = (g.cart + g.fav) / g.pv
    g = g.merge(cat_seq[["category_id", "cart2buy", "cart2buy_24h", "carted_pairs"]],
                on="category_id", how="left")
    g["view_share"] = g.pv / g.pv.sum()
    g["buy_share"] = g.buy / g.buy.sum()
    # 流量效率: 单位浏览产出购买
    g["efficiency"] = g.pv2buy / (g.buy.sum() / g.pv.sum())
    return g.sort_values("pv", ascending=False)


def by_time(df):
    hour = (df.groupby(["hour", "behavior_type"]).size().unstack(fill_value=0)
            .reindex(range(24)).fillna(0).reset_index())
    for col in VALID_TYPES:
        if col not in hour:
            hour[col] = 0
    hour["intent_rate"] = (hour.cart + hour.fav) / hour.pv
    hour["pv2buy"] = hour.buy / hour.pv
    hour["view_share"] = hour.pv / hour.pv.sum()
    hour["buy_share"] = hour.buy / hour.buy.sum()

    wd = (df.groupby(["weekday", "behavior_type"]).size().unstack(fill_value=0)
          .reindex(range(7)).fillna(0).reset_index())
    for col in VALID_TYPES:
        if col not in wd:
            wd[col] = 0
    wd["label"] = wd.weekday.map(dict(enumerate(WD_LABELS)))
    wd["intent_rate"] = (wd.cart + wd.fav) / wd.pv
    wd["pv2buy"] = wd.buy / wd.pv
    wd["view_share"] = wd.pv / wd.pv.sum()
    wd["buy_share"] = wd.buy / wd.buy.sum()
    return hour, wd


def by_time_bucket(df):
    """把 24 小时压成 4 个时段, 降低单小时样本波动。"""
    bins = [0, 7, 12, 18, 24]
    labels = ["凌晨 0-6", "上午 7-11", "下午 12-17", "晚间 18-23"]
    b = df.copy()
    b["bucket"] = pd.cut(b.hour, bins=bins, labels=labels, right=False, ordered=True)
    g = b.groupby(["bucket", "behavior_type"], observed=True).size().unstack(fill_value=0).reset_index()
    for col in VALID_TYPES:
        if col not in g:
            g[col] = 0
    g["intent_rate"] = (g.cart + g.fav) / g.pv
    g["pv2buy"] = g.buy / g.pv
    g["view_share"] = g.pv / g.pv.sum()
    g["buy_share"] = g.buy / g.buy.sum()
    return g


def by_user_segment(df):
    """按用户总浏览量的三分位切分活跃层级。"""
    u = (df.groupby(["user_id", "behavior_type"]).size().unstack(fill_value=0).reset_index())
    for col in VALID_TYPES:
        if col not in u:
            u[col] = 0
    q1, q2 = u.pv.quantile([1 / 3, 2 / 3])
    u["segment"] = np.select([u.pv <= q1, u.pv <= q2], ["低频", "中频"], default="高频")
    # 活跃天数
    days = df.assign(d=df.dt.dt.date).groupby("user_id").d.nunique()
    u["active_days"] = u.user_id.map(days)

    seg = (u.groupby("segment")
           .agg(users=("user_id", "count"),
                pv=("pv", "sum"), cart=("cart", "sum"), fav=("fav", "sum"), buy=("buy", "sum"),
                buyers=("buy", lambda s: int((s > 0).sum())),
                carts=("cart", lambda s: int((s > 0).sum())),
                median_pv=("pv", "median"),
                median_days=("active_days", "median"))
           .reset_index())
    seg["user_share"] = seg.users / seg.users.sum()
    seg["buy_user_rate"] = seg.buyers / seg.users
    seg["intent_user_rate"] = seg.carts / seg.users
    seg["pv2buy"] = seg.buy / seg.pv
    seg["orders_per_buyer"] = seg.buy / seg.buyers
    seg["pv_share"] = seg.pv / seg.pv.sum()
    seg["buy_share"] = seg.buy / seg.buy.sum()
    seg["pv_per_order"] = seg.pv / seg.buy
    order = ["低频", "中频", "高频"]
    seg["segment"] = pd.Categorical(seg.segment, order, ordered=True)
    return seg.sort_values("segment"), u


def category_anomaly(cat):
    """识别"高加购、低购买"类目: 加购率高于均值, 且加购→购买显著低于均值。"""
    mean_cart = cat.pv2cart.mean()
    mean_c2b = cat.cart2buy.mean()
    cat = cat.copy()
    cat["cart_gap"] = cat.pv2cart - mean_cart
    cat["c2b_gap"] = cat.cart2buy - mean_c2b
    high_cart_low_buy = cat[(cat.pv2cart > mean_cart) & (cat.cart2buy < mean_c2b)]
    return high_cart_low_buy.sort_values("c2b_gap"), mean_cart, mean_c2b


def main():
    OUT.mkdir(exist_ok=True)
    df, audit = load_and_clean()
    f, behavior = funnel(df)
    cat_seq, c2b_overall, conv_hour, conv_wd = cart_to_buy(df)
    cat = by_category(df, cat_seq)
    hour, wd = by_time(df)
    bucket = by_time_bucket(df)
    seg, _ = by_user_segment(df)
    anom, mean_cart, mean_c2b = category_anomaly(cat)

    f.to_csv(OUT / "funnel_users.csv", index=False, encoding="utf-8-sig")
    date_df = (df.assign(d=df.dt.dt.date)
               .groupby(["d", "behavior_type"]).size().unstack(fill_value=0).reset_index())
    date_df["intent_rate"] = (date_df.cart + date_df.fav) / date_df.pv
    date_df["pv2buy"] = date_df.buy / date_df.pv
    date_df.to_csv(OUT / "by_date.csv", index=False, encoding="utf-8-sig")
    bucket.to_csv(OUT / "by_time_bucket.csv", index=False, encoding="utf-8-sig")
    cat.to_csv(OUT / "by_category.csv", index=False, encoding="utf-8-sig")
    hour.to_csv(OUT / "by_hour.csv", index=False, encoding="utf-8-sig")
    wd.to_csv(OUT / "by_weekday.csv", index=False, encoding="utf-8-sig")
    seg.to_csv(OUT / "by_user_segment.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "audit": audit,
        "funnel": behavior,
        "funnel_users": f.to_dict("records"),
        "cart2buy": c2b_overall,
        "hour": {
            "peak_view_hour": int(hour.loc[hour.pv.idxmax(), "hour"]),
            "peak_buy_conv_hour": int(hour[hour.view_share >= 0.03].pipe(lambda d: d.loc[d.pv2buy.idxmax(), "hour"])),
            "worst_conv_hour": int(hour[hour.view_share >= 0.03].pipe(lambda d: d.loc[d.pv2buy.idxmin(), "hour"])),
            "conv_ratio_best_worst": float(hour[hour.view_share >= 0.03].pv2buy.max() / hour[hour.view_share >= 0.03].pv2buy.min()),
        },
        "weekday": {
            "best_conv": wd.loc[wd.pv2buy.idxmax(), "label"],
            "worst_conv": wd.loc[wd.pv2buy.idxmin(), "label"],
            "weekend_conv": float(wd[wd.weekday >= 5].pv2buy.mean()),
            "weekday_conv": float(wd[wd.weekday < 5].pv2buy.mean()),
            "weekend_view_share": float(wd[wd.weekday >= 5].pv.sum() / wd.pv.sum()),
        },
        "category": {
            "top_buy_conv": cat.nlargest(5, "pv2buy")[["category_id", "pv", "pv2cart", "cart2buy", "pv2buy"]].round(4).to_dict("records"),
            "anomaly_ids": anom.category_id.tolist()[:5],
            "anomaly": anom.head(5)[["category_id", "pv", "pv2cart", "cart2buy", "pv2buy", "cart_gap", "c2b_gap"]].round(4).to_dict("records"),
            "mean_pv2cart": float(mean_cart),
            "mean_cart2buy": float(mean_c2b),
        },
        "segment": seg.round(4).to_dict("records"),
        "bucket": bucket.round(4).to_dict("records"),
        "cart_hour_conv": {
            "best_hour": int(conv_hour.loc[conv_hour.cart2buy.idxmax(), "cart_hour"]),
            "best_rate": float(conv_hour.cart2buy.max()),
            "worst_hour": int(conv_hour.loc[conv_hour.cart2buy.idxmin(), "cart_hour"]),
            "worst_rate": float(conv_hour.cart2buy.min()),
            "night_rate": float(conv_hour[conv_hour.cart_hour < 7].converted.sum() /
                                conv_hour[conv_hour.cart_hour < 7].carts.sum()),
            "day_rate": float(conv_hour[(conv_hour.cart_hour >= 7) & (conv_hour.cart_hour < 18)].converted.sum() /
                              conv_hour[(conv_hour.cart_hour >= 7) & (conv_hour.cart_hour < 18)].carts.sum()),
            "evening_rate": float(conv_hour[conv_hour.cart_hour >= 18].converted.sum() /
                                  conv_hour[conv_hour.cart_hour >= 18].carts.sum()),
        },
        "cart_weekday_conv": {r["label"]: round(float(r["cart2buy"]), 4) for _, r in conv_wd.iterrows()},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=float),
                                      encoding="utf-8")

    # 控制台速览
    print(json.dumps({
        "rows": audit["clean_rows"], "users": audit["users"],
        "shares": {k: round(v / audit["clean_rows"], 4) for k, v in audit["behavior_counts"].items()},
        "funnel": behavior,
        "cart2buy": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c2b_overall.items()},
        "hour": metrics["hour"], "weekday": metrics["weekday"],
    }, ensure_ascii=False, indent=2))
    print("\n用户分层:\n", seg.round(3).to_string(index=False))
    print("\n高加购低购买类目:\n", anom.head(5)[["category_id", "pv", "pv2cart", "cart2buy", "pv2buy"]].round(3).to_string(index=False))
    print("\n转化最高类目:\n", cat.nlargest(5, "pv2buy")[["category_id", "pv", "pv2cart", "cart2buy", "pv2buy"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
