"""测试夹具：几行最小数据 + 内存 DuckDB，避免单测依赖 6.9MB 合成数据。"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import duckdb
import pandas as pd
import pytest

from uba.config import REPO_ROOT, get_settings
from uba.db import clean

T0 = 1_700_000_000  # 2023-11-15 06:13 (Asia/Shanghai)，任意基准秒


def _make_raw(rows: list[tuple[object, ...]]) -> pd.DataFrame:
    """rows: (user_id, item_id, category_id, behavior_type, timestamp)"""
    return pd.DataFrame(rows, columns=["user_id", "item_id", "category_id", "behavior_type", "timestamp"])


@pytest.fixture
def con():
    """每个测试一条独立的内存连接；单测不落盘。"""
    connection = duckdb.connect()
    connection.execute("SET threads=1")
    yield connection
    connection.close()


@pytest.fixture
def sample_settings(tmp_path):
    """窗口放宽到 2023 全年，让 T0 相对偏移（含 +200h 的删失用例）都落在窗口内。"""
    return replace(
        get_settings(),
        outputs_dir=tmp_path / "outputs",
        figures_dir=tmp_path / "figures",
        report_dir=tmp_path / "report",
        cache_dir=tmp_path / "cache",
        window_start=date(2023, 1, 1),
        window_end=date(2023, 12, 31),
    )


@pytest.fixture
def seed_events(con, sample_settings):
    """把几行原始事件注册成 `raw_events` 并跑清洗，返回审计字典。"""

    def _seed(rows: list[tuple[object, ...]]) -> dict[str, object]:
        con.execute("DROP VIEW IF EXISTS raw_events")
        con.register("raw_sample", _make_raw(rows))
        con.execute("CREATE OR REPLACE VIEW raw_events AS SELECT * FROM raw_sample")
        return clean(con, sample_settings)

    return _seed


@pytest.fixture
def tmp_settings(tmp_path):
    """默认窗口（2017-11-25~12-03），供合成数据/端到端测试使用。"""
    return replace(
        get_settings(),
        outputs_dir=tmp_path / "outputs",
        figures_dir=tmp_path / "figures",
        report_dir=tmp_path / "report",
        cache_dir=tmp_path / "cache",
        data_path=REPO_ROOT / "data" / "userbehavior.csv.gz",
    )
