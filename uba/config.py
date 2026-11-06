"""运行配置与全局常量。

路径与算力参数全部走环境变量：真实数据集（1.06 GB Parquet，1 亿行）不在仓库里，
复跑时位置随机器变化；`.env` / 环境变量覆盖后，换数据源不需要改代码。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 行为类型与展示顺序：公开数据集只定义这四类
BEHAVIOR_TYPES: tuple[str, ...] = ("pv", "cart", "fav", "buy")
WEEKDAY_LABELS: tuple[str, ...] = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
# 24 小时压成 4 段，降低单小时样本波动；左闭右开
TIME_BUCKETS: tuple[int, ...] = (0, 7, 12, 18, 24)
TIME_BUCKET_LABELS: tuple[str, ...] = ("凌晨 0-6", "上午 7-11", "下午 12-17", "晚间 18-23")

# 原始 timestamp 是 epoch 秒；公开数据集按北京时间解读（UTC+8）
TIMEZONE = "Asia/Shanghai"
# 数据集官方时间窗：窗口外的时间戳属于噪声/损坏，清洗阶段显式剔除并计数
WINDOW_START = date(2017, 11, 25)
WINDOW_END = date(2017, 12, 3)

# 观测窗口校正：只统计距数据末尾至少 48h 的购物车，避免右删失把末尾判成"未转化"
CENSOR_WINDOW_HOURS = 48

DEFAULT_SEED = 42
DEFAULT_MEMORY_LIMIT = "8GB"
# 1 亿行用 DuckDB 在磁盘上落表，内存只留给聚合；线程数默认留给机器一半
DEFAULT_THREADS = max(2, (os.cpu_count() or 4) // 2)


@dataclass(frozen=True)
class Settings:
    """一次运行所需的全部配置。"""

    root: Path
    data_path: Path
    outputs_dir: Path
    figures_dir: Path
    report_dir: Path
    cache_dir: Path
    log_level: str
    memory_limit: str
    threads: int
    censor_window_hours: int
    window_start: date
    window_end: date
    seed: int

    @property
    def report_markdown(self) -> Path:
        return self.report_dir / "attribution-report.md"

    @property
    def report_html(self) -> Path:
        return self.report_dir / "attribution-report.html"

    @property
    def database_path(self) -> Path:
        return self.cache_dir / "uba.duckdb"


def _load_dotenv(path: Path) -> None:
    """极简 .env 读取，避免为几个变量引入 python-dotenv。

    已存在的环境变量优先（CI / 临时覆盖不该被 .env 覆盖）。
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return default
    path = Path(raw).expanduser()
    # 相对路径按仓库根目录解析，避免"换个工作目录就跑不动"
    return path if path.is_absolute() else REPO_ROOT / path


def _date_from_env(name: str, default: date) -> date:
    raw = os.environ.get(name)
    return date.fromisoformat(raw) if raw else default


def get_settings() -> Settings:
    _load_dotenv(REPO_ROOT / ".env")
    return Settings(
        root=REPO_ROOT,
        data_path=_path_from_env("UBA_DATA_PATH", REPO_ROOT / "data" / "userbehavior.csv.gz"),
        outputs_dir=_path_from_env("UBA_OUTPUT_DIR", REPO_ROOT / "outputs"),
        figures_dir=_path_from_env("UBA_FIGURES_DIR", REPO_ROOT / "figures"),
        report_dir=_path_from_env("UBA_REPORT_DIR", REPO_ROOT / "report"),
        cache_dir=_path_from_env("UBA_CACHE_DIR", REPO_ROOT / "build"),
        log_level=os.environ.get("UBA_LOG_LEVEL", "INFO").upper(),
        memory_limit=os.environ.get("UBA_DUCKDB_MEMORY", DEFAULT_MEMORY_LIMIT),
        threads=int(os.environ.get("UBA_DUCKDB_THREADS", DEFAULT_THREADS)),
        censor_window_hours=int(os.environ.get("UBA_CENSOR_WINDOW_HOURS", CENSOR_WINDOW_HOURS)),
        window_start=_date_from_env("UBA_WINDOW_START", WINDOW_START),
        window_end=_date_from_env("UBA_WINDOW_END", WINDOW_END),
        seed=int(os.environ.get("UBA_SEED", DEFAULT_SEED)),
    )
