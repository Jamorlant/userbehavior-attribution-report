"""核心口径的单测：清洗、漏斗、序列转化、分层。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uba.analysis import (
    by_user_segment,
    cart_to_buy,
    category_anomaly,
    funnel,
    wilson_ci,
)

H = 3600
T0 = 1_700_000_000


def test_wilson_ci_normal_range():
    lo, hi = wilson_ci(50, 100)
    assert 0.39 < lo < 0.41
    assert 0.59 < hi < 0.61
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_funnel_users_level(con, seed_events):
    seed_events(
        [
            (1, 101, 1, "pv", T0),
            (1, 101, 1, "cart", T0 + 10),
            (1, 101, 1, "buy", T0 + 20),
            (2, 102, 1, "pv", T0 + 30),
            (2, 102, 1, "fav", T0 + 40),
            (3, 103, 1, "pv", T0 + 50),
        ]
    )
    table, behavior = funnel(con)

    rates = dict(zip(table.step, table.rate, strict=True))
    assert rates["意向 (加购∪收藏)"] == pytest.approx(2 / 3)
    assert rates["购买 (buy)"] == pytest.approx(1 / 2)
    assert behavior["pv2buy_users"] == pytest.approx(1 / 3)
    assert behavior["intent2buy_users"] == pytest.approx(1 / 2)
    assert behavior["pv2buy"] == pytest.approx(1 / 3)  # 行为级：1 buy / 3 pv


def test_cart_to_buy_sequence_and_censoring(con, seed_events):
    """序列口径：加购之后的购买才算转化；末尾 48h 内的购物车剔除。"""
    t_max = 200 * H
    seed_events(
        [
            (1, 11, 1, "pv", T0),
            (1, 11, 1, "cart", T0 + 100),
            (1, 11, 1, "buy", T0 + 200),  # 加购后购买 → 转化
            (2, 12, 2, "buy", T0),
            (2, 12, 2, "cart", T0 + 100),  # 先买后加购 → 不算转化
            (3, 13, 3, "cart", T0 + 100),
            (3, 13, 3, "buy", T0 + 100 + 25 * H),  # 25h 后购买 → 转化，但不算 24h
            (4, 14, 4, "cart", T0 + t_max - 1800),  # 窗口末尾 → 删失
            (5, 15, 5, "buy", T0 + 100),  # 未经加购直接下单
            (9, 19, 9, "pv", T0 + t_max),
        ]
    )
    result = cart_to_buy(con)

    assert result.overall["censored_pairs"] == 1
    assert result.overall["carted_pairs"] == 3
    assert result.overall["converted_pairs"] == 2
    assert result.overall["cart2buy"] == pytest.approx(2 / 3)
    assert result.overall["cart2buy_24h"] == pytest.approx(1 / 3)
    assert result.overall["median_lag_hours"] == pytest.approx((100 / 3600 + 25) / 2)
    assert result.overall["buy_share_without_cart"] == pytest.approx(1 / 4)

    # 时段归因按"加购小时"，不是下单小时
    assert result.by_hour["carts"].sum() == 3
    assert result.by_weekday["carts"].sum() == 3


def test_cart_to_buy_without_conversions_returns_nan(con, seed_events):
    seed_events(
        [
            (1, 11, 1, "cart", T0),
            (2, 12, 1, "pv", T0 + 200 * H),
        ]
    )
    result = cart_to_buy(con)
    assert result.overall["converted_pairs"] == 0
    assert np.isnan(result.overall["median_lag_hours"])


def test_clean_dedupes_and_filters_explicitly(seed_events):
    audit = seed_events(
        [
            (1, 11, 1, "pv", T0),  # 正常
            (1, 11, 1, "pv", T0),  # 完全重复
            (2, 12, 1, "click", T0 + 10),  # 非法行为类型
            (3, 13, 1, "pv", T0 + 20),  # 正常
            (4, 14, 1, "cart", T0 + 30),  # 正常
            (None, 15, 1, "pv", T0 + 40),  # 空 user_id
            (5, 15, 1, "pv", 0),  # 1970 年 → 窗口外 + 损坏时间戳
        ]
    )

    assert audit["clean_rows"] == 3
    assert audit["dup_rows_removed"] == 1
    assert audit["out_of_window_rows"] == 1
    assert audit["corrupt_timestamp_rows"] == 1
    assert audit["invalid_rows"] == 2
    assert audit["days"] == 1


def test_clean_rejects_empty(con, sample_settings):
    from uba.db import clean

    con.execute("DROP VIEW IF EXISTS raw_events")
    con.register(
        "raw_sample",
        pd.DataFrame(
            [(1, 1, 1, "click", T0)], columns=["user_id", "item_id", "category_id", "behavior_type", "timestamp"]
        ),
    )
    con.execute("CREATE OR REPLACE VIEW raw_events AS SELECT * FROM raw_sample")
    with pytest.raises(ValueError, match="清洗后没有剩余数据"):
        clean(con, sample_settings)


def test_by_user_segment_quantile_split(con, seed_events):
    rows: list[tuple[object, ...]] = []
    for user, n_pv in ((1, 1), (2, 4), (3, 9)):
        rows += [(user, 100 + user, 1, "pv", T0 + i) for i in range(n_pv)]
        rows.append((user, 100 + user, 1, "buy", T0 + 100 + n_pv))
    seed_events(rows)
    seg = by_user_segment(con)

    assert list(seg.segment) == ["低频", "中频", "高频"]
    assert seg.user_share.sum() == pytest.approx(1.0)
    assert seg.pv_share.sum() == pytest.approx(1.0)


def test_category_anomaly_flags_high_cart_low_buy():
    cat = pd.DataFrame(
        {
            "category_id": [1, 2, 3],
            "pv": [10_000, 10_000, 10_000],
            "carted_pairs": [600, 600, 600],
            "pv2cart": [0.20, 0.05, 0.10],
            "cart2buy": [0.05, 0.40, 0.30],
            "pv2buy": [0.010, 0.020, 0.030],
        }
    )
    result = category_anomaly(cat)
    assert result.categories.category_id.tolist() == [1]
    assert result.mean_pv2cart == pytest.approx((0.20 + 0.05 + 0.10) / 3)
