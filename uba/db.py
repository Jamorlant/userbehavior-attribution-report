"""数据接入与清洗：Parquet/CSV → DuckDB 表 `events`。

为什么不在 pandas 里读 1 亿行：
    UserBehavior 原始 CSV 约 3.5 GB，pandas 默认 dtype 读入后常驻内存 6 GB 以上，
    再做 groupby/merge 很容易在 16 GB 机器上 OOM。DuckDB 用磁盘落表 + 列式聚合，
    内存峰值只与聚合结果有关，同一套 SQL 也能在几万行的小样本上跑（单测即如此）。
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from uba.config import BEHAVIOR_TYPES, TIMEZONE, Settings
from uba.logging_utils import get_logger

log = get_logger("uba.db")

REQUIRED_COLUMNS = ("user_id", "item_id", "category_id", "behavior_type", "timestamp")

_SOURCE_SQL = "SELECT user_id, item_id, category_id, behavior_type, timestamp FROM {scan}"


def _scan_expr(path: Path) -> str:
    """按后缀选择扫描函数；路径用单引号包裹（DuckDB 的 read_* 接受文件路径字面量）。"""
    suffix = path.suffix.lower()
    quoted = "'" + path.as_posix().replace("'", "''") + "'"
    if suffix == ".parquet":
        return f"read_parquet({quoted})"
    if suffix in {".csv", ".gz", ".txt"}:
        return f"read_csv({quoted}, header=true)"
    raise ValueError(f"不支持的数据格式: {path.suffix}（支持 .parquet / .csv / .csv.gz）")


def connect(settings: Settings) -> duckdb.DuckDBPyConnection:
    """打开磁盘数据库连接。

    表落在 cache_dir/uba.duckdb：1 亿行事件表约 4 GB，放内存会挤掉聚合所需的 working set。
    """
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(settings.database_path))
    con.execute(f"SET memory_limit='{settings.memory_limit}'")
    con.execute(f"SET threads={settings.threads}")
    con.execute(f"SET temp_directory='{settings.cache_dir.as_posix().replace(chr(39), chr(39) * 2)}'")
    # 事件表不依赖插入顺序；关掉保序可显著降低大表构建时的内存
    con.execute("SET preserve_insertion_order=false")
    return con


def register_source(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    """把原始文件注册成视图 `raw_events`，并校验 schema。返回原始行数。"""
    if not path.is_file():
        raise FileNotFoundError(
            f"数据文件不存在: {path}。用 UBA_DATA_PATH 指向真实数据集，"
            "或先运行 `python -m uba.synthetic` 生成合成数据。"
        )
    con.execute(f"CREATE OR REPLACE VIEW raw_events AS {_SOURCE_SQL.format(scan=_scan_expr(path))}")
    columns = {row[0] for row in con.execute("DESCRIBE raw_events").fetchall()}
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"缺少必需字段 {missing}；实际字段: {sorted(columns)}")
    rows = int(con.execute("SELECT count(*) FROM raw_events").fetchone()[0])
    if rows == 0:
        raise ValueError(f"数据文件没有数据行: {path}")
    log.info("已载入原始数据 %s（%s 行）", path.name, f"{rows:,}")
    return rows


def clean(con: duckdb.DuckDBPyConnection, settings: Settings) -> dict[str, object]:
    """去空/非法行为、去重、按官方时间窗过滤，落成表 `events`，返回审计字典。

    时间戳按北京时间（UTC+8）解读；窗口外的时间戳（含 1902/2037 等损坏值）一律剔除，
    剔除数量在这里显式计数，不允许"悄悄删行"。
    """
    start, end = settings.window_start.isoformat(), settings.window_end.isoformat()
    local_ts = f"to_timestamp(timestamp) AT TIME ZONE '{TIMEZONE}'"
    valid = f"behavior_type IN ({', '.join(repr(t) for t in BEHAVIOR_TYPES)})"
    cleanable = (
        f"{valid} AND user_id IS NOT NULL AND item_id IS NOT NULL AND category_id IS NOT NULL AND timestamp IS NOT NULL"
    )
    in_window = f"CAST({local_ts} AS DATE) BETWEEN DATE '{start}' AND DATE '{end}'"

    raw_rows = int(con.execute("SELECT count(*) FROM raw_events").fetchone()[0])
    proxies = con.execute(
        f"""
        SELECT
            count(*) FILTER (WHERE {cleanable} AND {in_window}) AS valid_in_window,
            count(*) FILTER (WHERE CAST({local_ts} AS DATE) NOT BETWEEN DATE '{start}' AND DATE '{end}') AS out_window,
            count(*) FILTER (WHERE year({local_ts}) <> {settings.window_start.year}) AS corrupt,
            count(*) FILTER (WHERE NOT ({cleanable})) AS invalid
        FROM raw_events
        """
    ).fetchone()
    valid_in_window, out_window, corrupt, invalid = (int(v) for v in proxies)

    con.execute(
        f"""
        CREATE OR REPLACE TABLE events AS
        SELECT DISTINCT
            user_id,
            item_id,
            category_id,
            behavior_type,
            timestamp,
            CAST({local_ts} AS DATE) AS d,
            CAST(EXTRACT(hour FROM {local_ts}) AS TINYINT) AS hour,
            CAST(EXTRACT(isodow FROM {local_ts}) - 1 AS TINYINT) AS weekday
        FROM raw_events
        WHERE {cleanable} AND {in_window}
        """
    )

    clean_rows = int(con.execute("SELECT count(*) FROM events").fetchone()[0])
    if clean_rows == 0:
        raise ValueError("清洗后没有剩余数据，请检查字段与时间戳格式")

    stats = con.execute(
        """
        SELECT count(DISTINCT user_id) AS users,
               count(DISTINCT item_id) AS items,
               count(DISTINCT category_id) AS categories,
               count(DISTINCT d) AS days,
               min(d) AS date_start,
               max(d) AS date_end,
               min(timestamp) AS ts_start,
               max(timestamp) AS ts_end
        FROM events
        """
    ).fetchone()
    behavior = dict(con.execute("SELECT behavior_type, count(*) FROM events GROUP BY 1").fetchall())

    audit: dict[str, object] = {
        "raw_rows": raw_rows,
        "clean_rows": clean_rows,
        "dropped_rows": raw_rows - clean_rows,
        "dup_rows_removed": valid_in_window - clean_rows,
        "out_of_window_rows": out_window,
        "corrupt_timestamp_rows": corrupt,
        "invalid_rows": invalid,
        "users": int(stats[0]),
        "items": int(stats[1]),
        "categories": int(stats[2]),
        "days": int(stats[3]),
        "date_start": str(stats[4]),
        "date_end": str(stats[5]),
        "window_start": start,
        "window_end": end,
        "timezone": TIMEZONE,
        "behavior_counts": {k: int(v) for k, v in behavior.items()},
    }
    log.info(
        "清洗完成: 保留 %s 行 / %s 用户 / %s 商品 / %s 类目 / %s 天",
        f"{clean_rows:,}",
        f"{audit['users']:,}",
        f"{audit['items']:,}",
        f"{audit['categories']:,}",
        audit["days"],
    )
    log.info(
        "剔除明细: 窗口外 %s 行（其中时间戳年份异常 %s 行）、重复 %s 行、非法/空值 %s 行",
        f"{out_window:,}",
        f"{corrupt:,}",
        f"{audit['dup_rows_removed']:,}",
        f"{invalid:,}",
    )
    return audit
