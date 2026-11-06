"""漏斗与归因指标（SQL 口径，结果统一返回小尺寸 DataFrame）。

口径约定（也是本报告与"看起来像分析"的区别）：
    行为级转化率 = 目标行为次数 / pv 次数
    加购→购买（序列口径）= 同一 (user_id, item_id) 上，加购之后是否出现购买
    观测窗校正：距数据末尾不足 48h 的购物车剔除，否则末尾必然被判为未转化
    时段归因：按"加购发生的小时"统计，而不是下单小时（下单延迟中位约数小时）

所有函数都只读 DuckDB 表 `events`（由 uba.db 清洗产出），不写盘，便于单测直接
用几行小样本注册成同名表验证口径。
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from uba.config import (
    BEHAVIOR_TYPES,
    TIME_BUCKET_LABELS,
    TIMEZONE,
    WEEKDAY_LABELS,
)

EVENTS = "events"

BUCKET_SQL = (
    "CASE "
    "WHEN hour < 7 THEN '凌晨 0-6' "
    "WHEN hour < 12 THEN '上午 7-11' "
    "WHEN hour < 18 THEN '下午 12-17' "
    "ELSE '晚间 18-23' END"
)


@dataclass(frozen=True)
class CartConversion:
    by_category: pd.DataFrame
    overall: dict[str, float | int]
    by_hour: pd.DataFrame
    by_weekday: pd.DataFrame
    lag_hist: pd.DataFrame


@dataclass(frozen=True)
class CategoryAnomaly:
    categories: pd.DataFrame
    mean_pv2cart: float
    mean_cart2buy: float


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 置信区间，小样本下比正态近似稳。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return float((c - r) / d), float((c + r) / d)


def funnel(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """用户级漏斗：浏览 → 意向(加购∪收藏) → 购买，附 Wilson 95% CI。"""
    counts = dict(con.execute(f"SELECT behavior_type, count(*) FROM {EVENTS} GROUP BY 1").fetchall())
    missing = [b for b in BEHAVIOR_TYPES if b not in counts]
    if missing:
        raise ValueError(f"缺少行为类型 {missing}，无法构建漏斗")

    users = con.execute(
        f"""
        SELECT count(DISTINCT user_id) FILTER (WHERE behavior_type = 'pv') AS pv_users,
               count(DISTINCT user_id) FILTER (WHERE behavior_type = 'cart') AS cart_users,
               count(DISTINCT user_id) FILTER (WHERE behavior_type = 'fav') AS fav_users,
               count(DISTINCT user_id) FILTER (WHERE behavior_type = 'buy') AS buy_users
        FROM {EVENTS}
        """
    ).fetchone()
    intent_users = int(
        con.execute(f"SELECT count(DISTINCT user_id) FROM {EVENTS} WHERE behavior_type IN ('cart','fav')").fetchone()[0]
    )
    u_pv, u_cart, u_fav, u_buy = (int(v) for v in users)
    if u_pv == 0:
        raise ValueError("没有 pv 行为，无法构建漏斗")

    steps = [
        ("浏览 (pv)", u_pv, None),
        ("加购 (cart)", u_cart, u_pv),
        ("收藏 (fav)", u_fav, u_pv),
        ("意向 (加购∪收藏)", intent_users, u_pv),
        ("购买 (buy)", u_buy, intent_users),
    ]
    rows: list[dict[str, object]] = []
    for name, n, base in steps:
        rate = (n / base) if base else np.nan
        lo, hi = wilson_ci(n, base) if base else (np.nan, np.nan)
        rows.append(dict(step=name, users=n, base_users=base, rate=rate, ci_low=lo, ci_high=hi))

    pv = int(counts["pv"])
    behavior: dict[str, float | int] = {
        "pv": pv,
        "cart": int(counts["cart"]),
        "fav": int(counts["fav"]),
        "buy": int(counts["buy"]),
        "pv2cart": float(counts["cart"] / pv),
        "pv2fav": float(counts["fav"] / pv),
        "pv2intent": float((counts["cart"] + counts["fav"]) / pv),
        "pv2buy": float(counts["buy"] / pv),
        "intent2buy_users": float(u_buy / intent_users),
        "pv2buy_users": float(u_buy / u_pv),
    }
    return pd.DataFrame(rows), behavior


def build_cart_pairs(con: duckdb.DuckDBPyConnection, censor_hours: int) -> None:
    """构建序列口径的 (user,item) 加购对表（含是否在观测窗内）。

    只保留"加购之后"的购买：加购前就买过的商品不算这次加购的转化。
    """
    con.execute(
        f"""
        CREATE OR REPLACE TABLE cart_pairs AS
        WITH carts AS (
            SELECT user_id, item_id, min(timestamp) AS cart_ts,
                   arg_min(category_id, timestamp) AS category_id
            FROM {EVENTS}
            WHERE behavior_type = 'cart'
            GROUP BY 1, 2
        ),
        buys AS (
            SELECT user_id, item_id, min(timestamp) AS buy_ts
            FROM {EVENTS}
            WHERE behavior_type = 'buy'
            GROUP BY 1, 2
        ),
        tmax AS (SELECT max(timestamp) AS ts FROM {EVENTS})
        SELECT c.user_id, c.item_id, c.category_id, c.cart_ts, b.buy_ts,
               (b.buy_ts IS NOT NULL AND b.buy_ts >= c.cart_ts) AS converted,
               (b.buy_ts IS NOT NULL AND b.buy_ts >= c.cart_ts
                AND b.buy_ts - c.cart_ts <= 86400) AS converted_24h,
               CASE WHEN b.buy_ts IS NOT NULL AND b.buy_ts >= c.cart_ts
                    THEN b.buy_ts - c.cart_ts END AS lag_seconds,
               c.cart_ts <= tmax.ts - {censor_hours * 3600} AS observed
        FROM carts c
        LEFT JOIN buys b ON c.user_id = b.user_id AND c.item_id = b.item_id
        CROSS JOIN tmax
        """
    )


def cart_to_buy(con: duckdb.DuckDBPyConnection, censor_hours: int = 48) -> CartConversion:
    """序列口径的加购→购买，并按加购时段/类目拆分。"""
    build_cart_pairs(con, censor_hours)

    overall_row = con.execute(
        """
        SELECT count(*) AS carted_pairs,
               count(*) FILTER (WHERE converted) AS converted_pairs,
               avg(CASE WHEN converted THEN 1.0 ELSE 0.0 END) AS cart2buy,
               avg(CASE WHEN converted_24h THEN 1.0 ELSE 0.0 END) AS cart2buy_24h,
               median(lag_seconds) / 3600.0 AS median_lag_hours,
               avg(CASE WHEN lag_seconds <= 3600 THEN 1.0 WHEN lag_seconds IS NOT NULL THEN 0.0 END)
                   AS converted_within_1h,
               avg(CASE WHEN lag_seconds <= 86400 THEN 1.0 WHEN lag_seconds IS NOT NULL THEN 0.0 END)
                   AS converted_within_24h
        FROM cart_pairs WHERE observed
        """
    ).fetchone()
    censored = int(con.execute("SELECT count(*) FROM cart_pairs WHERE NOT observed").fetchone()[0])

    def _num(value: float | int | None) -> float:
        """没有已成交样本时 SQL 聚合返回 NULL，指标里用 NaN 表达"样本不足"。"""
        return float("nan") if value is None else float(value)

    overall: dict[str, float | int] = {
        "carted_pairs": int(overall_row[0]),
        "converted_pairs": int(overall_row[1]),
        "cart2buy": _num(overall_row[2]),
        "cart2buy_24h": _num(overall_row[3]),
        "median_lag_hours": _num(overall_row[4]),
        "converted_within_1h": _num(overall_row[5]),
        "converted_within_24h": _num(overall_row[6]),
        "censored_pairs": censored,
    }

    # 直接下单（没有任何加购记录）的订单占比：说明加购不是唯一入口
    buy_events, buys_with_cart = con.execute(
        f"""
        SELECT (SELECT count(*) FROM {EVENTS} WHERE behavior_type = 'buy') AS buy_events,
               (SELECT count(*) FROM {EVENTS} e WHERE e.behavior_type = 'buy'
                AND EXISTS (SELECT 1 FROM cart_pairs p
                            WHERE p.user_id = e.user_id AND p.item_id = e.item_id)) AS buys_with_cart
        """
    ).fetchone()
    overall["buy_events"] = int(buy_events)
    overall["buy_share_without_cart"] = float(1 - buys_with_cart / buy_events) if buy_events else float("nan")

    by_cat = con.execute(
        """
        SELECT category_id,
               count(*) AS carted_pairs,
               count(*) FILTER (WHERE converted) AS converted,
               count(*) FILTER (WHERE converted_24h) AS converted_24h
        FROM cart_pairs WHERE observed
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    by_cat["cart2buy"] = by_cat["converted"] / by_cat["carted_pairs"]
    by_cat["cart2buy_24h"] = by_cat["converted_24h"] / by_cat["carted_pairs"]

    local_cart = f"to_timestamp(cart_ts) AT TIME ZONE '{TIMEZONE}'"
    by_hour = con.execute(
        f"""
        SELECT CAST(EXTRACT(hour FROM {local_cart}) AS INTEGER) AS cart_hour,
               count(*) AS carts,
               count(*) FILTER (WHERE converted) AS converted
        FROM cart_pairs WHERE observed
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    by_hour = by_hour.set_index("cart_hour").reindex(range(24), fill_value=0).reset_index()
    by_hour["cart2buy"] = by_hour["converted"] / by_hour["carts"].replace(0, np.nan)
    by_hour["cart_share"] = by_hour["carts"] / by_hour["carts"].sum()

    by_weekday = con.execute(
        f"""
        SELECT CAST(EXTRACT(isodow FROM {local_cart}) - 1 AS INTEGER) AS cart_weekday,
               count(*) AS carts,
               count(*) FILTER (WHERE converted) AS converted
        FROM cart_pairs WHERE observed
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    by_weekday = by_weekday.set_index("cart_weekday").reindex(range(7), fill_value=0).reset_index()
    by_weekday["label"] = by_weekday["cart_weekday"].map(dict(enumerate(WEEKDAY_LABELS)))
    by_weekday["cart2buy"] = by_weekday["converted"] / by_weekday["carts"].replace(0, np.nan)

    # 成交延迟直方图：按 1 小时分箱，避免把上百万行明细落盘
    lag_hist = con.execute(
        """
        SELECT CAST(floor(lag_seconds / 3600) AS INTEGER) AS bin_hours, count(*) AS carts
        FROM cart_pairs WHERE observed AND converted
        GROUP BY 1 ORDER BY 1
        """
    ).df()

    return CartConversion(
        by_category=by_cat,
        overall=overall,
        by_hour=by_hour,
        by_weekday=by_weekday,
        lag_hist=lag_hist,
    )


def by_category(con: duckdb.DuckDBPyConnection, cat_seq: pd.DataFrame) -> pd.DataFrame:
    """类目维度：行为量、行为级转化率、序列口径成交率、流量效率。"""
    g = con.execute(
        f"""
        SELECT category_id,
               count(*) FILTER (WHERE behavior_type = 'pv') AS pv,
               count(*) FILTER (WHERE behavior_type = 'cart') AS cart,
               count(*) FILTER (WHERE behavior_type = 'fav') AS fav,
               count(*) FILTER (WHERE behavior_type = 'buy') AS buy
        FROM {EVENTS}
        GROUP BY 1
        """
    ).df()
    g["pv2cart"] = g["cart"] / g["pv"].replace(0, np.nan)
    g["pv2fav"] = g["fav"] / g["pv"].replace(0, np.nan)
    g["pv2buy"] = g["buy"] / g["pv"].replace(0, np.nan)
    g["intent_rate"] = (g["cart"] + g["fav"]) / g["pv"].replace(0, np.nan)
    g = g.merge(cat_seq[["category_id", "cart2buy", "cart2buy_24h", "carted_pairs"]], on="category_id", how="left")
    g["view_share"] = g["pv"] / g["pv"].sum()
    g["buy_share"] = g["buy"] / g["buy"].sum()
    # 流量效率 = 类目转化率 / 全站转化率，>1 表示单位流量产出高于大盘
    g["efficiency"] = g["pv2buy"] / (g["buy"].sum() / g["pv"].sum())
    # 按 (pv, category_id) 双键排序：只有 pv 排序时，同浏览量的类目顺序随扫描顺序变化，
    # 会导致 CSV / metrics.json 每次重跑字节不一致（无法用指纹验证"结果没变"）
    return g.sort_values(["pv", "category_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def _pivot(con: duckdb.DuckDBPyConnection, key_sql: str, order_sql: str) -> pd.DataFrame:
    """按 key 聚合四种行为次数，并补齐转化率列。"""
    out = con.execute(
        f"""
        SELECT {key_sql} AS key,
               count(*) FILTER (WHERE behavior_type = 'pv') AS pv,
               count(*) FILTER (WHERE behavior_type = 'cart') AS cart,
               count(*) FILTER (WHERE behavior_type = 'fav') AS fav,
               count(*) FILTER (WHERE behavior_type = 'buy') AS buy
        FROM {EVENTS}
        GROUP BY 1 ORDER BY {order_sql}
        """
    ).df()
    out["intent_rate"] = (out["cart"] + out["fav"]) / out["pv"]
    out["pv2buy"] = out["buy"] / out["pv"]
    out["view_share"] = out["pv"] / out["pv"].sum()
    out["buy_share"] = out["buy"] / out["buy"].sum()
    return out


def by_date(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return _pivot(con, "d", "key").rename(columns={"key": "d"})


def by_time(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    """小时 / 星期两套指标，列含义一致。"""
    hour = _pivot(con, "hour", "key").rename(columns={"key": "hour"})
    wd = _pivot(con, "weekday", "key").rename(columns={"key": "weekday"})
    wd.insert(5, "label", wd["weekday"].map(dict(enumerate(WEEKDAY_LABELS))))
    return hour, wd


def by_time_bucket(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """把 24 小时压成 4 个时段，降低单小时样本波动。"""
    out = _pivot(con, BUCKET_SQL, "min(hour)").rename(columns={"key": "bucket"})
    out["bucket"] = pd.Categorical(out["bucket"], list(TIME_BUCKET_LABELS), ordered=True)
    return out.sort_values("bucket").reset_index(drop=True)


def by_user_segment(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """按用户总浏览量三分位切分活跃层级（低频/中频/高频）。"""
    seg = con.execute(
        f"""
        WITH u AS (
            SELECT user_id,
                   count(*) FILTER (WHERE behavior_type = 'pv') AS pv,
                   count(*) FILTER (WHERE behavior_type = 'cart') AS cart,
                   count(*) FILTER (WHERE behavior_type = 'fav') AS fav,
                   count(*) FILTER (WHERE behavior_type = 'buy') AS buy,
                   count(DISTINCT d) AS active_days
            FROM {EVENTS}
            GROUP BY 1
        ),
        q AS (SELECT quantile_cont(pv, 1.0 / 3) AS q1, quantile_cont(pv, 2.0 / 3) AS q2 FROM u)
        SELECT CASE WHEN u.pv <= q.q1 THEN '低频'
                    WHEN u.pv <= q.q2 THEN '中频'
                    ELSE '高频' END AS segment,
               count(*) AS users,
               sum(pv) AS pv, sum(cart) AS cart, sum(fav) AS fav, sum(buy) AS buy,
               count(*) FILTER (WHERE buy > 0) AS buyers,
               count(*) FILTER (WHERE cart > 0) AS carts,
               median(pv) AS median_pv,
               median(active_days) AS median_days
        FROM u CROSS JOIN q
        GROUP BY 1
        """
    ).df()
    seg["user_share"] = seg["users"] / seg["users"].sum()
    seg["buy_user_rate"] = seg["buyers"] / seg["users"]
    seg["intent_user_rate"] = seg["carts"] / seg["users"]
    seg["pv2buy"] = seg["buy"] / seg["pv"]
    seg["orders_per_buyer"] = seg["buy"] / seg["buyers"]
    seg["pv_share"] = seg["pv"] / seg["pv"].sum()
    seg["buy_share"] = seg["buy"] / seg["buy"].sum()
    seg["pv_per_order"] = seg["pv"] / seg["buy"]
    order = ["低频", "中频", "高频"]
    seg["segment"] = pd.Categorical(seg["segment"], order, ordered=True)
    return seg.sort_values("segment").reset_index(drop=True)


def category_anomaly(cat: pd.DataFrame, min_pv: int = 10_000, min_carts: int = 500) -> CategoryAnomaly:
    """识别"高加购、低成交"类目：加购率高于均值，且加购→购买显著低于均值。

    真实数据集有 9,437 个类目，其中不少只有几十次浏览（转化率波动极大），
    因此先按浏览量 / 购物车数设门槛，只评估有业务意义的类目。
    """
    sample = cat[(cat["pv"] >= min_pv) & (cat["carted_pairs"] >= min_carts)]
    if sample.empty:
        sample = cat[(cat["pv"] > 0) & (cat["carted_pairs"] > 0)]
    mean_cart = float(sample["pv2cart"].mean())
    mean_c2b = float(sample["cart2buy"].mean())
    flagged = sample.copy()
    flagged["cart_gap"] = flagged["pv2cart"] - mean_cart
    flagged["c2b_gap"] = flagged["cart2buy"] - mean_c2b
    picked = flagged[(flagged["pv2cart"] > mean_cart) & (flagged["cart2buy"] < mean_c2b)]
    return CategoryAnomaly(
        # 双键排序保证同 gap 的类目顺序稳定，metrics.json 才可复现
        categories=picked.sort_values(["c2b_gap", "category_id"], kind="mergesort"),
        mean_pv2cart=mean_cart,
        mean_cart2buy=mean_c2b,
    )
