"""
生成与阿里巴巴天池「淘宝用户行为数据集 UserBehavior」同 schema 的合成数据。

字段: user_id,item_id,category_id,behavior_type,timestamp
      behavior_type ∈ {pv, cart, fav, buy}

说明:
  本脚本不下载任何真实用户数据。它按公开数据集的统计特征(行为占比、昼夜节律、
  类目长尾、加购→购买条件概率)建模, 生成可复现的模拟数据, 仅用于验证分析方法。
  所有 user_id 均为随机整数, 不与任何自然人对应。

用法:
    python scripts/generate_data.py --rows 1000000 --out data/userbehavior.csv.gz
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_USERS = 58_000
N_ITEMS = 6_200
N_CATEGORIES = 40
START = int(pd.Timestamp("2017-11-25 00:00:00").value // 10**9)  # 复刻公开数据集的时间窗口起点
N_DAYS = 9

# 类目原型: 刻画不同品类的决策路径
#   pv2cart 浏览→加购率; cart2buy 加购→购买率; pv2fav 浏览→收藏率
ARCHETYPES = {
    "impulse":    dict(pv2cart=0.115, cart2buy=0.40, pv2fav=0.030),  # 低客单冲动型
    "mainstream": dict(pv2cart=0.078, cart2buy=0.30, pv2fav=0.028),  # 主流品类
    "deliberate": dict(pv2cart=0.052, cart2buy=0.18, pv2fav=0.042),  # 高客单决策型
    "compare":    dict(pv2cart=0.128, cart2buy=0.10, pv2fav=0.048),  # 比价/凑单型: 高加购低购买
    "search":     dict(pv2cart=0.028, cart2buy=0.58, pv2fav=0.010),  # 明确意图型: 低加购高购买
}
ARCH_COUNTS = {"impulse": 13, "mainstream": 15, "deliberate": 7, "compare": 3, "search": 2}

# 一天内浏览量的相对强度 (0-23 点)
HOUR_WEIGHT = np.array([
    0.34, 0.20, 0.12, 0.07, 0.05, 0.06, 0.14, 0.32, 0.55,
    0.66, 0.72, 0.76, 0.80, 0.84, 0.88, 0.94, 1.00, 1.10,
    1.30, 1.52, 1.72, 1.80, 1.48, 0.78,
])
# 下单转化的小时修正: 晚间决策更果断, 凌晨与清晨更弱
HOUR_CONV = np.array([
    0.52, 0.48, 0.46, 0.45, 0.45, 0.48, 0.62, 0.78, 0.92,
    1.02, 1.06, 1.05, 1.00, 0.98, 0.96, 0.97, 1.02, 1.08,
    1.16, 1.26, 1.34, 1.32, 1.22, 0.86,
])
# 周末浏览更旺但"逛而不买"
DAY_WEIGHT = np.array([1.10, 1.12, 0.92, 0.90, 0.95, 0.98, 1.02, 1.12, 1.06])
DAY_CONV = np.array([0.84, 0.86, 1.06, 1.08, 1.06, 1.05, 1.02, 0.83, 0.85])

# 用户活跃分层: 会话强度 / 购买意图强度
SEGMENTS = {
    "heavy": dict(share=0.15, lam=10.4, intent=0.78),
    "mid":   dict(share=0.35, lam=4.2, intent=1.00),
    "light": dict(share=0.50, lam=1.45, intent=1.36),
}


def build_catalog(rng):
    """类目与商品目录: 类目走长尾, 商品在类目内走长尾。"""
    arch_names = []
    for name, cnt in ARCH_COUNTS.items():
        arch_names += [name] * cnt
    arch_names = list(rng.permutation(np.array(arch_names)))

    pop = 1.0 / np.arange(1, N_CATEGORIES + 1) ** 0.85
    pop = pop / pop.sum()
    cat_pop = rng.permutation(pop)

    cat_id = np.arange(1, N_CATEGORIES + 1) * 137 + 11
    cat2arch = dict(zip(cat_id, arch_names))

    item_cat = rng.choice(cat_id, size=N_ITEMS, p=cat_pop)
    # 商品自身热度: 类目内长尾
    item_w = rng.gamma(1.6, 1.0, size=N_ITEMS) * np.array(
        [cat_pop[np.where(cat_id == c)[0][0]] for c in item_cat]
    ) ** 0.35
    item_w = item_w / item_w.sum()
    item_id = rng.choice(np.arange(10**6, 10**6 + N_ITEMS * 7, 7), size=N_ITEMS, replace=False)
    return dict(cat_id=cat_id, cat2arch=cat2arch, item_id=item_id, item_cat=item_cat,
                item_w=item_w, cat_pop=cat_pop, arch=ARCHETYPES)


def generate(rows_target, rng):
    catalog = build_catalog(rng)

    # ---------- 用户 ----------
    seg_names = list(SEGMENTS.keys())
    seg_share = np.array([SEGMENTS[s]["share"] for s in seg_names])
    user_seg = rng.choice(len(seg_names), size=N_USERS, p=seg_share)
    user_id = rng.choice(np.arange(10**8, 10**9), size=N_USERS, replace=False)

    lam = np.array([SEGMENTS[s]["lam"] for s in seg_names])[user_seg]
    intent = np.array([SEGMENTS[s]["intent"] for s in seg_names])[user_seg]
    # 个体异质性
    intent = intent * rng.gamma(4.0, 0.25, size=N_USERS)

    sessions_per_user = rng.poisson(lam)
    total_sessions = sessions_per_user.sum()
    # 事件量估算后对会话数做一次比例校准
    est_events = total_sessions * 5.0
    scale = rows_target / max(est_events, 1)
    if 0.85 < scale < 1.2:
        sessions_per_user = rng.binomial(sessions_per_user, min(scale, 1.0)) \
            if scale < 1 else sessions_per_user + rng.binomial(1, scale - 1, size=N_USERS)

    s_user = np.repeat(np.arange(N_USERS), sessions_per_user)
    n_sessions = len(s_user)

    # ---------- 会话时间 ----------
    day = rng.choice(N_DAYS, size=n_sessions, p=DAY_WEIGHT / DAY_WEIGHT.sum())
    hour_p = HOUR_WEIGHT / HOUR_WEIGHT.sum()
    hour = rng.choice(24, size=n_sessions, p=hour_p)
    t = (START + day * 86400 + hour * 3600 +
         rng.integers(0, 3600, size=n_sessions)).astype(np.int64)

    # ---------- 会话内的商品浏览 ----------
    n_items = np.clip(1 + rng.poisson(3.1, size=n_sessions), 1, 12)
    sess_item = np.repeat(np.arange(n_sessions), n_items)
    item_idx = rng.choice(N_ITEMS, size=len(sess_item), p=catalog["item_w"])

    u = s_user[sess_item]
    cat = catalog["item_cat"][item_idx]
    ts = t[sess_item]
    hh = hour[sess_item]
    dd = day[sess_item]

    pv2cart = np.array([ARCHETYPES[catalog["cat2arch"][c]]["pv2cart"] for c in catalog["cat_id"]])
    pv2fav = np.array([ARCHETYPES[catalog["cat2arch"][c]]["pv2fav"] for c in catalog["cat_id"]])
    cart2buy = np.array([ARCHETYPES[catalog["cat2arch"][c]]["cart2buy"] for c in catalog["cat_id"]])

    cat_pos = np.searchsorted(catalog["cat_id"], cat)
    conv_mult = HOUR_CONV[hh] * DAY_CONV[dd]
    # 晚间/周末的"意向行为"强度: 加购在晚间更容易发生
    intent_mult = (0.72 + 0.55 * (HOUR_CONV[hh] / HOUR_CONV.max())) * intent[u]

    is_cart = rng.random(len(u)) < pv2cart[cat_pos] * 0.55 * intent_mult
    is_fav = (rng.random(len(u)) < pv2fav[cat_pos] * 0.75 * (0.5 + intent_mult)) & ~is_cart
    # 直接下单(未加购), 占少数
    direct_buy = (rng.random(len(u)) < pv2cart[cat_pos] * 0.020 * intent_mult) & ~is_cart & ~is_fav
    buy_from_cart = is_cart & (rng.random(len(u)) < cart2buy[cat_pos] * conv_mult)
    buy_from_fav = is_fav & (rng.random(len(u)) < 0.10 * conv_mult)

    frames = []
    frames.append(pd.DataFrame({
        "user_id": user_id[u], "item_id": catalog["item_id"][item_idx],
        "category_id": cat, "behavior_type": "pv", "timestamp": ts,
    }))
    for flag, name in ((is_cart, "cart"), (is_fav, "fav")):
        frames.append(pd.DataFrame({
            "user_id": user_id[u][flag], "item_id": catalog["item_id"][item_idx][flag],
            "category_id": cat[flag], "behavior_type": name, "timestamp": ts[flag],
        }))
    # 购买事件: 加购后延迟下单(含跨天犹豫期) / 收藏后下单 / 直接下单
    # 超出观测窗口的下单视为"未转化"(右删失), 不做时间裁剪, 避免在窗口末尾堆积假峰值
    end = START + N_DAYS * 86400
    for flag, delay in (
        (buy_from_cart, np.clip(rng.exponential(43200, size=int(np.count_nonzero(buy_from_cart))), 60, 172800)),
        (buy_from_fav, np.clip(rng.exponential(28800, size=int(np.count_nonzero(buy_from_fav))), 900, 172800)),
        (direct_buy, rng.integers(30, 1800, size=int(np.count_nonzero(direct_buy))).astype(float)),
    ):
        idx = np.flatnonzero(flag)
        tt = ts[idx] + delay.astype(np.int64)
        keep = tt < end
        idx, tt = idx[keep], tt[keep]
        frames.append(pd.DataFrame({
            "user_id": user_id[u][idx], "item_id": catalog["item_id"][item_idx][idx],
            "category_id": cat[idx], "behavior_type": "buy", "timestamp": tt,
        }))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["user_id"] = df["user_id"].astype(np.int64)
    df["item_id"] = df["item_id"].astype(np.int64)
    df["category_id"] = df["category_id"].astype(np.int64)
    df["timestamp"] = df["timestamp"].astype(np.int64)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--out", default="data/userbehavior.csv.gz")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    df = generate(args.rows, rng)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, compression="gzip")

    stats = {
        "rows": int(len(df)),
        "users": int(df.user_id.nunique()),
        "items": int(df.item_id.nunique()),
        "categories": int(df.category_id.nunique()),
        "behavior_share": (df.behavior_type.value_counts(normalize=True).round(4)).to_dict(),
        "time_range": [str(pd.to_datetime(df.timestamp.min(), unit="s")),
                       str(pd.to_datetime(df.timestamp.max(), unit="s"))],
        "seed": SEED,
        "file_size_mb": round(out.stat().st_size / 2**20, 2),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
