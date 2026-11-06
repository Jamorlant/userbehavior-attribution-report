"""端到端冒烟：小样本数据 → 指标落盘，验证编排层没有断点。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from uba.logging_utils import configure_logging
from uba.pipeline import run
from uba.synthetic import generate

EXPECTED_CSV = [
    "funnel_users.csv",
    "by_date.csv",
    "by_time_bucket.csv",
    "by_category.csv",
    "by_hour.csv",
    "by_weekday.csv",
    "by_user_segment.csv",
    "cart_conv_by_hour.csv",
    "cart_conv_by_weekday.csv",
    "cart_lag_hist.csv",
]


def test_pipeline_end_to_end(tmp_settings, tmp_path):
    df = generate(20_000, np.random.default_rng(7))
    data_path = tmp_path / "sample.csv.gz"
    df.to_csv(data_path, index=False, compression="gzip")

    settings = replace(tmp_settings, data_path=data_path, log_level="ERROR")
    configure_logging("ERROR")
    metrics = run(settings)

    assert settings.outputs_dir.is_dir()
    assert (settings.outputs_dir / "metrics.json").is_file()
    for name in EXPECTED_CSV:
        assert (settings.outputs_dir / name).is_file(), name

    # 生成器可能产生完全重复事件（同一 user/item/type/timestamp），清洗会去重
    assert metrics["audit"]["clean_rows"] == len(df.drop_duplicates())
    assert metrics["audit"]["dropped_rows"] == len(df) - metrics["audit"]["clean_rows"]
    assert metrics["funnel"]["pv"] > 0
    assert 0 < metrics["funnel"]["pv2buy"] < 0.2
    assert len(metrics["funnel_users"]) == 5
