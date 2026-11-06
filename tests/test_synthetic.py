"""合成数据生成器的契约：schema、可复现、行数可控。"""

from __future__ import annotations

import numpy as np
import pytest

from uba.config import BEHAVIOR_TYPES
from uba.synthetic import generate

ROWS = 5_000


@pytest.fixture(scope="module")
def sample():
    return generate(ROWS, np.random.default_rng(42))


def test_schema_matches_public_dataset(sample):
    assert list(sample.columns) == ["user_id", "item_id", "category_id", "behavior_type", "timestamp"]
    assert set(sample.behavior_type) <= set(BEHAVIOR_TYPES)
    assert all(str(sample[c].dtype) == "int64" for c in ("user_id", "item_id", "category_id", "timestamp"))


def test_rows_target_is_respected_at_small_scale(sample):
    # 校准允许 ±25% 偏差：请求 5,000 行不应该返回 90 万行
    assert 3_500 <= len(sample) <= 6_500


def test_sorted_by_timestamp(sample):
    assert sample.timestamp.is_monotonic_increasing


def test_same_seed_is_reproducible():
    a = generate(ROWS, np.random.default_rng(42))
    b = generate(ROWS, np.random.default_rng(42))
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
