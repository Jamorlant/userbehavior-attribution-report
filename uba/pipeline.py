"""分析管道：接入数据 → 计算指标 → 落盘 outputs/。

入口：`python -m uba.pipeline [--data PATH] [--out DIR]`
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import duckdb
import pandas as pd

from uba.analysis import (
    CartConversion,
    by_category,
    by_date,
    by_time,
    by_time_bucket,
    by_user_segment,
    cart_to_buy,
    category_anomaly,
    funnel,
)
from uba.config import Settings, get_settings
from uba.db import clean, connect, register_source
from uba.logging_utils import configure_logging, get_logger

log = get_logger("uba.pipeline")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    # utf-8-sig：Excel 直接打开不乱码（分析师实际会这么用）
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_metrics(
    audit: dict[str, object],
    behavior: dict[str, float | int],
    funnel_df: pd.DataFrame,
    c2b: dict[str, float | int],
    hour: pd.DataFrame,
    wd: pd.DataFrame,
    cat: pd.DataFrame,
    anomaly: pd.DataFrame,
    mean_cart: float,
    mean_c2b: float,
    seg: pd.DataFrame,
    bucket: pd.DataFrame,
    conv_hour: pd.DataFrame,
    conv_wd: pd.DataFrame,
    source_file: str,
) -> dict[str, object]:
    """汇总 metrics.json —— 报告与 README 的每个数字都应能在这里溯源。"""
    meaningful_hours = hour[hour["view_share"] >= 0.03]
    return {
        "source": {"file": source_file, "window": [audit["window_start"], audit["window_end"]]},
        "audit": audit,
        "funnel": behavior,
        "funnel_users": funnel_df.to_dict("records"),
        "cart2buy": c2b,
        "hour": {
            "peak_view_hour": int(hour.loc[hour["pv"].idxmax(), "hour"]),
            "peak_buy_conv_hour": int(meaningful_hours.loc[meaningful_hours["pv2buy"].idxmax(), "hour"]),
            "worst_conv_hour": int(meaningful_hours.loc[meaningful_hours["pv2buy"].idxmin(), "hour"]),
            "conv_ratio_best_worst": float(meaningful_hours["pv2buy"].max() / meaningful_hours["pv2buy"].min()),
        },
        "weekday": {
            "best_conv": str(wd.loc[wd["pv2buy"].idxmax(), "label"]),
            "worst_conv": str(wd.loc[wd["pv2buy"].idxmin(), "label"]),
            "weekend_conv": float(wd[wd["weekday"] >= 5]["pv2buy"].mean()),
            "weekday_conv": float(wd[wd["weekday"] < 5]["pv2buy"].mean()),
            "weekend_view_share": float(wd[wd["weekday"] >= 5]["pv"].sum() / wd["pv"].sum()),
        },
        "category": {
            "top_buy_conv": cat[(cat["pv"] >= 10_000) & (cat["cart"] > 0)]
            .nlargest(5, "pv2buy")[["category_id", "pv", "pv2cart", "cart2buy", "pv2buy"]]
            .round(4)
            .to_dict("records"),
            "anomaly_ids": anomaly["category_id"].head(5).tolist(),
            "anomaly": anomaly.head(5)[["category_id", "pv", "pv2cart", "cart2buy", "pv2buy", "cart_gap", "c2b_gap"]]
            .round(4)
            .to_dict("records"),
            "mean_pv2cart": float(mean_cart),
            "mean_cart2buy": float(mean_c2b),
        },
        "segment": seg.round(4).to_dict("records"),
        "bucket": bucket.round(4).to_dict("records"),
        "cart_hour_conv": {
            "best_hour": int(conv_hour.loc[conv_hour["cart2buy"].idxmax(), "cart_hour"]),
            "best_rate": float(conv_hour["cart2buy"].max()),
            "worst_hour": int(conv_hour.loc[conv_hour["cart2buy"].idxmin(), "cart_hour"]),
            "worst_rate": float(conv_hour["cart2buy"].min()),
            "night_rate": float(
                conv_hour[conv_hour["cart_hour"] < 7]["converted"].sum()
                / conv_hour[conv_hour["cart_hour"] < 7]["carts"].sum()
            ),
            "day_rate": float(
                conv_hour[(conv_hour["cart_hour"] >= 7) & (conv_hour["cart_hour"] < 18)]["converted"].sum()
                / conv_hour[(conv_hour["cart_hour"] >= 7) & (conv_hour["cart_hour"] < 18)]["carts"].sum()
            ),
            "evening_rate": float(
                conv_hour[conv_hour["cart_hour"] >= 18]["converted"].sum()
                / conv_hour[conv_hour["cart_hour"] >= 18]["carts"].sum()
            ),
        },
        "cart_weekday_conv": {r["label"]: round(float(r["cart2buy"]), 4) for _, r in conv_wd.iterrows()},
    }


def run(settings: Settings) -> dict[str, object]:
    con = connect(settings)
    try:
        register_source(con, settings.data_path)
        audit = clean(con, settings)
        funnel_df, behavior = funnel(con)
        conv = cart_to_buy(con, settings.censor_window_hours)
        log.info(
            "加购→购买 %.2f%%（48h 窗；24h 内完成 %.1f%%），删失购物车 %s 个",
            conv.overall["cart2buy"] * 100,
            conv.overall["converted_within_24h"] * 100,
            f"{conv.overall['censored_pairs']:,}",
        )
        cat = by_category(con, conv.by_category)
        hour, wd = by_time(con)
        bucket = by_time_bucket(con)
        seg = by_user_segment(con)
        daily = by_date(con)
        anomaly = category_anomaly(cat)

        metrics = build_metrics(
            audit=audit,
            behavior=behavior,
            funnel_df=funnel_df,
            c2b=conv.overall,
            hour=hour,
            wd=wd,
            cat=cat,
            anomaly=anomaly.categories,
            mean_cart=anomaly.mean_pv2cart,
            mean_c2b=anomaly.mean_cart2buy,
            seg=seg,
            bucket=bucket,
            conv_hour=conv.by_hour,
            conv_wd=conv.by_weekday,
            source_file=settings.data_path.name,
        )
        _write_outputs(settings, funnel_df, daily, bucket, cat, hour, wd, seg, conv, metrics)
    finally:
        con.close()

    _log_summary(behavior, conv.overall, metrics, seg, anomaly.categories, cat)
    return metrics


def _write_outputs(
    settings: Settings,
    funnel_df: pd.DataFrame,
    daily: pd.DataFrame,
    bucket: pd.DataFrame,
    cat: pd.DataFrame,
    hour: pd.DataFrame,
    wd: pd.DataFrame,
    seg: pd.DataFrame,
    conv: CartConversion,
    metrics: dict[str, object],
) -> None:
    out = settings.outputs_dir
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(funnel_df, out / "funnel_users.csv")
    _write_csv(daily, out / "by_date.csv")
    _write_csv(bucket, out / "by_time_bucket.csv")
    _write_csv(cat, out / "by_category.csv")
    _write_csv(hour, out / "by_hour.csv")
    _write_csv(wd, out / "by_weekday.csv")
    _write_csv(seg, out / "by_user_segment.csv")
    _write_csv(conv.by_hour, out / "cart_conv_by_hour.csv")
    _write_csv(conv.by_weekday, out / "cart_conv_by_weekday.csv")
    _write_csv(conv.lag_hist, out / "cart_lag_hist.csv")
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    log.info("指标已写入: %s", out)


def _log_summary(
    behavior: dict[str, float | int],
    c2b: dict[str, float | int],
    metrics: dict[str, object],
    seg: pd.DataFrame,
    anomaly: pd.DataFrame,
    cat: pd.DataFrame,
) -> None:
    """控制台速览：只放结论级数字，明细在 outputs/。"""
    total = sum(int(behavior[k]) for k in ("pv", "cart", "fav", "buy"))
    shares = {k: round(int(behavior[k]) / total, 4) for k in ("pv", "cart", "fav", "buy")}
    log.info("行为占比: %s", json.dumps(shares, ensure_ascii=False))
    log.info("漏斗: %s", json.dumps(behavior, ensure_ascii=False))
    log.info("小时/星期: %s", json.dumps({"hour": metrics["hour"], "weekday": metrics["weekday"]}, ensure_ascii=False))
    log.info("用户分层:\n%s", seg.round(3).to_string(index=False))
    log.info(
        "高加购低购买类目:\n%s",
        anomaly.head(5)[["category_id", "pv", "pv2cart", "cart2buy", "pv2buy"]].round(3).to_string(index=False),
    )
    log.info(
        "转化最高类目:\n%s",
        cat[(cat["pv"] >= 10_000) & (cat["cart"] > 0)]
        .nlargest(5, "pv2buy")[["category_id", "pv", "pv2cart", "cart2buy", "pv2buy"]]
        .round(3)
        .to_string(index=False),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="计算漏斗与归因指标，输出到 outputs/")
    ap.add_argument("--data", type=Path, help="原始数据路径（.parquet / .csv / .csv.gz）")
    ap.add_argument("--out", type=Path, help="指标输出目录")
    ap.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
    args = ap.parse_args(argv)

    settings = get_settings()
    overrides: dict[str, object] = {}
    if args.data:
        overrides["data_path"] = args.data
    if args.out:
        overrides["outputs_dir"] = args.out
    if args.log_level:
        overrides["log_level"] = args.log_level
    if overrides:
        settings = replace(settings, **overrides)

    configure_logging(settings.log_level)
    try:
        run(settings)
    except (FileNotFoundError, ValueError, duckdb.Error) as exc:
        log.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
