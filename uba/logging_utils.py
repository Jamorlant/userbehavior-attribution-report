"""统一日志配置。

为什么不用 print：管道有"读数据 / 清洗 / 指标 / 落盘"多段，需要级别区分——
    DEBUG 轨迹、INFO 里程碑、WARNING 数据质量异常、ERROR 无法继续。
"""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_FORMAT,
        datefmt="%H:%M:%S",
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
