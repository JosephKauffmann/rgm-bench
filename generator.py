"""Synthetic weekly POS data for the Liftbench demo (v3).

Everything is invented. The demand process is a textbook multiplicative model with the
standard RGM structure (Van Heerde & Neslin, "Sales Promotion Models"): a baseline moved by
BASE price elasticity (list-price changes), seasonality, holiday and PRE-holiday build; and
a promotional response driven by a separate PROMO price elasticity plus merchandising
mechanics (display, feature, coupon), with pre-promo deal anticipation and post-promo
pantry-loading dips, sibling cannibalization, and a 5% rate of execution failures.
Ground truth is kept alongside so recovery can be proved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(11)

N_WEEKS = 156
WEEKS = pd.date_range(end="2026-09-05", periods=N_WEEKS, freq="W-SAT")

# name, terminal_stores, velocity_mult, price_tier, list-increase weeks (staggered by banner
# so base elasticity is identified separately from the common trend)
RETAILERS = [
    ("Fernhollow Market",    110, 0.9, "premium", (22, 78, 118)),
    ("Granite Peak Grocers",  95, 1.0, "mid",     (28, 86, 126)),
    ("ThriftLane",           140, 1.5, "value",   (36, 96, 134)),
    ("Hollybrook Foods",      45, 0.8, "premium", (25, 82, 112)),
]

PRICE_LADDER = {
    "BW-101": {"premium": 5.49, "mid": 4.99, "value": 4.29},
    "BW-102": {"premium": 5.49, "mid": 4.99, "value": 4.29},
    "BW-103": {"premium": 6.99, "mid": 6.49, "value": 5.49},
    "BW-201": {"premium": 4.49, "mid": 3.99, "value": 3.49},
    "BW-202": {"premium": 4.49, "mid": 3.99, "value": 3.29},
    "BW-203": {"premium": 7.99, "mid": 7.29, "value": 6.49},
}
CHARM = np.array([0.29, 0.49, 0.79, 0.99])
def charm_round(p: float) -> float:
    base = np.floor(p)
    cands = np.concatenate([base - 1 + CHARM, base + CHARM])
    return float(cands[np.argmin(np.abs(cands - p))])

# sku_id, name, line, cogs, BASE price elasticity, PROMO price elasticity,
# display_mult, feature_mult, coupon_mult, season
SKUS = [
    ("BW-101", "Bramblewick Granola Clusters Honey 12oz", "Granola", 1.32, -1.3, -2.0, 1.32, 1.20, 1.12, "winter"),
    ("BW-102", "Bramblewick Granola Clusters Cocoa 12oz", "Granola", 1.36, -1.1, -1.7, 1.28, 1.17, 1.10, "winter"),
    ("BW-103", "Bramblewick Granola Clusters Maple 18oz", "Granola", 1.74, -0.9, -1.5, 1.25, 1.15, 1.09, "winter"),
    ("BW-201", "Bramblewick Trail Bars Peanut 6ct",       "Bars",    1.02, -1.6, -2.2, 1.38, 1.24, 1.14, "summer"),
    ("BW-202", "Bramblewick Trail Bars Berry 6ct",        "Bars",    1.23, -1.4, -2.1, 1.34, 1.21, 1.12, "summer"),
    ("BW-203", "Bramblewick Trail Bars Dark Choc 12ct",   "Bars",    2.04, -1.0, -1.6, 1.28, 1.16, 1.10, "summer"),
]
BASE_VELOCITY = {"BW-101": 5.2, "BW-102": 4.4, "BW-103": 2.9, "BW-201": 6.0, "BW-202": 5.1, "BW-203": 2.4}

CANNIBALIZATION = 0.88
PRE_PROMO_DIP = 0.96     # deal anticipation the week before an event
POST_PROMO_DIP = 0.93    # pantry loading the week after
WHOLESALE_SHARE = 0.66
LAUNCH = {("BW-203", "Hollybrook Foods"): 40}
COUPON_FACE = 0.75       # $ per redemption
COUPON_REDEMPTION = 0.08 # share of promo units redeeming

HOLIDAYS = {
    "newyear":      ["2024-01-06", "2024-01-13", "2025-01-04", "2025-01-11", "2026-01-03", "2026-01-10"],
    "thanksgiving": ["2023-11-25", "2024-11-30", "2025-11-29"],
    "christmas":    ["2023-12-23", "2023-12-30", "2024-12-21", "2024-12-28", "2025-12-20", "2025-12-27"],
    "july4":        ["2024-07-06", "2025-07-05", "2026-07-04"],
    "memorial":     ["2024-05-25", "2025-05-31", "2026-05-30"],
    "backtoschool": ["2023-09-16", "2024-08-24", "2024-08-31", "2025-08-23", "2025-08-30", "2026-08-22", "2026-08-29"],
}
# shopping happens BEFORE the occasion: pre-holiday build weeks carry their own bump
PRE_BUILD = {"thanksgiving": 0.14, "christmas": 0.12, "july4": 0.16, "memorial": 0.10}
HOLIDAY_BUMPS = {"newyear": 0.20, "thanksgiving": 0.20, "christmas": 0.15, "july4": 0.25, "memorial": 0.15, "backtoschool": 0.12}
HOLIDAY_LINE_SCALE = {
    ("Granola", "newyear"): 1.5, ("Granola", "thanksgiving"): 1.2, ("Granola", "christmas"): 1.1,
    ("Granola", "july4"): 0.3, ("Granola", "memorial"): 0.3, ("Granola", "backtoschool"): 1.2,
    ("Bars", "newyear"): 0.6, ("Bars", "thanksgiving"): 0.4, ("Bars", "christmas"): 0.3,
    ("Bars", "july4"): 1.5, ("Bars", "memorial"): 1.4, ("Bars", "backtoschool"): 1.3,
}

def matrix_for(dates_map) -> pd.DataFrame:
    m = pd.DataFrame(0.0, index=WEEKS, columns=list(dates_map))
    for name, dates in dates_map.items():
        for d in dates:
            d = pd.Timestamp(d)
            if d in m.index:
                m.loc[d, name] = 1.0
    return m

HOL = matrix_for(HOLIDAYS)
PRE = matrix_for({h: [str((pd.Timestamp(d) - pd.Timedelta(weeks=1)).date()) for d in HOLIDAYS[h]]
                  for h in PRE_BUILD})

T = np.arange(N_WEEKS)
WOY = WEEKS.isocalendar().week.to_numpy().astype(float)
AD_WEEKS = sorted({WEEKS.get_loc(pd.Timestamp(d)) for ds in HOLIDAYS.values() for d in ds if pd.Timestamp(d) in WEEKS})
BANNER_CADENCE = {"Fernhollow Market": 0.7, "Granite Peak Grocers": 1.0, "ThriftLane": 1.4, "Hollybrook Foods": 0.6}

def everyday_series(sku_id: str, tier: str, increase_weeks: tuple) -> np.ndarray:
    p0 = PRICE_LADDER[sku_id][tier]
    p = np.full(N_WEEKS, p0)
    for wk, m in zip(increase_weeks, (1.07, 1.06, 1.05)):
        p[T >= wk] = charm_round(float(p[T >= wk][0]) * m)
    return p


def build_promo_calendar() -> pd.DataFrame:
    """Tiered trade plan: TPR-only / ad-supported / hero; coupons ride ad & hero events."""
    events, eid = [], 0
    for sku in SKUS:
        sku_id = sku[0]
        for rname, stores, vel, tier, incw in RETAILERS:
            launch = LAUNCH.get((sku_id, rname), 0)
            occupied = np.zeros(N_WEEKS, dtype=bool)
            n_events = int(round(RNG.integers(15, 20) * BANNER_CADENCE[rname]))
            placed, attempts = 0, 0
            while placed < n_events and attempts < 400:
                attempts += 1
                tier_kind = RNG.choice(["tpr", "ad", "hero"], p=[0.45, 0.35, 0.20])
                if tier_kind == "tpr":
                    depth = float(RNG.choice([0.10, 0.15])); display = False; feature = False
                    coupon = False; dur = int(RNG.choice([1, 2]))
                elif tier_kind == "ad":
                    depth = float(RNG.choice([0.20, 0.25])); display = RNG.random() < 0.35; feature = True
                    coupon = bool(RNG.random() < 0.25); dur = int(RNG.choice([1, 2]))
                else:
                    depth = float(RNG.choice([0.25, 0.30])); display = True; feature = True
                    coupon = bool(RNG.random() < 0.35); dur = int(RNG.choice([2, 3]))
                if tier == "value" and depth > 0.25: depth = 0.20
                if tier == "premium" and tier_kind == "tpr" and RNG.random() < 0.4: depth = 0.15
                if tier_kind != "tpr" and RNG.random() < 0.6 and AD_WEEKS:
                    start = int(RNG.choice(AD_WEEKS)) - int(RNG.integers(0, dur))
                    start = max(launch + 2, min(start, N_WEEKS - dur - 2))
                else:
                    start = int(RNG.integers(max(2, launch + 2), N_WEEKS - dur - 2))
                guard = slice(max(0, start - 2), min(N_WEEKS, start + dur + 2))
                if occupied[guard].any():
                    continue
                occupied[slice(start, start + dur)] = True
                events.append(dict(
                    event_id=f"EV-{eid:04d}", sku_id=sku_id, retailer=rname,
                    start_week=str(WEEKS[start].date()), end_week=str(WEEKS[start + dur - 1].date()),
                    start_idx=start, dur=dur, depth=depth, display=bool(display), feature=bool(feature),
                    coupon=coupon, tier_kind=tier_kind,
                    funding_type=str(RNG.choice(["scanback", "off_invoice"], p=[0.7, 0.3])),
                    exec_fail=bool(RNG.random() < 0.05),
                ))
                eid += 1
                placed += 1
    return pd.DataFrame(events)


def stores_path(terminal: int, launch: int) -> np.ndarray:
    start = int(terminal * RNG.uniform(0.55, 0.75))
    steps = np.linspace(start, terminal, 12)
    path = np.zeros(N_WEEKS)
    for i in range(12):
        path[i * 13:(i + 1) * 13] = int(steps[i])
    path = path + RNG.integers(-1, 2, N_WEEKS)
    if launch:
        path[:launch] = 0
    return np.clip(path, 0, None)


def simulate() -> tuple[pd.DataFrame, pd.DataFrame]:
    cal = build_promo_calendar()
    prep = []
    for sku_id, name, line, cogs, base_pe, promo_pe, dmult, fmult, cmult, season in SKUS:
        amp1 = 0.16 if season == "winter" else 0.20
        phase = np.pi / 2 - 2 * np.pi * (1.5 / 52) if season == "winter" else np.pi / 2 - 2 * np.pi * (27 / 52)
        season_ln = amp1 * np.sin(2 * np.pi * WOY / 52 + phase) + 0.05 * np.sin(4 * np.pi * WOY / 52)
        hol_ln = np.zeros(N_WEEKS)
        for h in HOLIDAYS:
            hol_ln += HOL[h].to_numpy() * HOLIDAY_BUMPS[h] * HOLIDAY_LINE_SCALE[(line, h)]
        for h in PRE_BUILD:
            hol_ln += PRE[h].to_numpy() * PRE_BUILD[h] * HOLIDAY_LINE_SCALE[(line, h)]
        for rname, terminal, vel_mult, tier, incw in RETAILERS:
            launch = LAUNCH.get((sku_id, rname), 0)
            stores = stores_path(terminal, launch)
            everyday = everyday_series(sku_id, tier, incw)
            velocity = BASE_VELOCITY[sku_id] * vel_mult
            trend = RNG.normal(0.0006, 0.0005)
            pair_events = cal[(cal.sku_id == sku_id) & (cal.retailer == rname)]
            depth_w = np.zeros(N_WEEKS); disp_w = np.zeros(N_WEEKS); feat_w = np.zeros(N_WEEKS)
            coup_w = np.zeros(N_WEEKS); fail_w = np.zeros(N_WEEKS, dtype=bool)
            pre_w = np.zeros(N_WEEKS, dtype=bool); post_w = np.zeros(N_WEEKS, dtype=bool)
            for _, ev in pair_events.iterrows():
                w = slice(ev.start_idx, ev.start_idx + ev.dur)
                depth_w[w] = ev.depth; disp_w[w] = float(ev.display); feat_w[w] = float(ev.feature)
                coup_w[w] = float(ev.coupon); fail_w[w] = ev.exec_fail
                if ev.start_idx >= 1:
                    pre_w[ev.start_idx - 1] = True
                if ev.start_idx + ev.dur < N_WEEKS:
                    post_w[ev.start_idx + ev.dur] = True
            prep.append(dict(
                sku_id=sku_id, name=name, line=line, retailer=rname, cogs=cogs,
                base_pe=base_pe, promo_pe=promo_pe, dmult=dmult, fmult=fmult, cmult=cmult,
                season_ln=season_ln, hol_ln=hol_ln, trend=trend, stores=stores,
                everyday=everyday, velocity=velocity, launch=launch,
                depth_w=depth_w, disp_w=disp_w, feat_w=feat_w, coup_w=coup_w,
                fail_w=fail_w, pre_w=pre_w, post_w=post_w,
            ))

    weekly_rows = []
    for r in prep:
        sibs = [s for s in prep if s["line"] == r["line"] and s["retailer"] == r["retailer"] and s["sku_id"] != r["sku_id"]]
        sib_promo = np.zeros(N_WEEKS)
        for s in sibs:
            sib_promo = np.maximum(sib_promo, (s["depth_w"] > 0).astype(float))

        # BASE price elasticity: baseline responds to list-price changes (vs launch price)
        ln_list_rel = np.log(r["everyday"] / r["everyday"][0])
        ln_base = (np.log(np.maximum(r["velocity"] * r["stores"], 1e-9))
                   + r["trend"] * T + r["season_ln"] + r["hol_ln"]
                   + r["base_pe"] * ln_list_rel)
        base_true = np.where(r["stores"] > 0, np.exp(ln_base), 0.0)

        price = np.where(r["depth_w"] > 0, r["everyday"] * (1 - r["depth_w"]), r["everyday"]).round(2)
        eff_ratio = np.where(r["fail_w"], 1.0, price / r["everyday"])
        lift_ln = (r["promo_pe"] * np.log(eff_ratio)
                   + np.log(r["dmult"]) * np.where(r["fail_w"], 0, r["disp_w"])
                   + np.log(r["fmult"]) * np.where(r["fail_w"], 0, r["feat_w"])
                   + np.log(r["cmult"]) * np.where(r["fail_w"], 0, r["coup_w"])
                   + np.log(CANNIBALIZATION) * sib_promo * (r["depth_w"] == 0)
                   + np.log(PRE_PROMO_DIP) * r["pre_w"].astype(float)
                   + np.log(POST_PROMO_DIP) * r["post_w"].astype(float))
        sigma = np.where(r["depth_w"] > 0, 0.14, 0.11)
        units = np.where(r["stores"] > 0, np.exp(ln_base + lift_ln + RNG.normal(0, sigma)), 0.0)

        n_oos = RNG.integers(0, 3)
        oos = np.zeros(N_WEEKS, dtype=bool)
        for _ in range(n_oos):
            wk_i = int(RNG.integers(max(4, r["launch"] + 4), N_WEEKS - 4))
            if r["depth_w"][wk_i] == 0:
                units[wk_i] *= RNG.uniform(0.25, 0.5)
                oos[wk_i] = True

        part = np.where(r["depth_w"] > 0, RNG.uniform(0.75, 0.97, N_WEEKS), 1.0)
        arp = (price * part + r["everyday"] * (1 - part)).round(2)
        pct_acv = np.clip(r["stores"] / max(r["stores"].max(), 1) * RNG.uniform(0.92, 1.0), 0, 1)

        df = pd.DataFrame(dict(
            week_end=WEEKS.strftime("%Y-%m-%d"), t=T,
            sku_id=r["sku_id"], sku_name=r["name"], line=r["line"], retailer=r["retailer"],
            units=np.round(units, 1), everyday_price=r["everyday"], price=price,
            dollars=np.round(units * arp, 2), arp=arp,
            promo_flag=(r["depth_w"] > 0).astype(int), depth=r["depth_w"],
            display=r["disp_w"].astype(int), feature=r["feat_w"].astype(int), coupon=r["coup_w"].astype(int),
            sibling_promo=sib_promo.astype(int), pre_promo=r["pre_w"].astype(int), post_promo=r["post_w"].astype(int),
            stores_selling=r["stores"].astype(int), pct_acv=np.round(pct_acv, 3), oos_week=oos.astype(int),
            base_units_true=np.round(base_true, 1), cogs=r["cogs"],
            true_base_pe=r["base_pe"], true_promo_pe=r["promo_pe"],
            true_display_mult=r["dmult"], true_feature_mult=r["fmult"], true_coupon_mult=r["cmult"],
        ))
        weekly_rows.append(df[df.stores_selling > 0])
    weekly = pd.concat(weekly_rows, ignore_index=True).sort_values(["sku_id", "retailer", "t"]).reset_index(drop=True)

    # naive trailing-median baseline: an ILLUSTRATIVE reference method, not any vendor's algorithm
    def naive_base(g: pd.DataFrame) -> pd.Series:
        vals, hist = [], []
        for _, row in g.iterrows():
            est = float(np.median(hist[-8:])) if len(hist) >= 4 else row.units
            vals.append(est if row.promo_flag else row.units)
            if not row.promo_flag and not row.oos_week:
                hist.append(row.units)
        return pd.Series(vals, index=g.index)

    weekly["naive_base_units"] = weekly.groupby(["sku_id", "retailer"], group_keys=False).apply(naive_base, include_groups=False)
    weekly["naive_incr_units"] = np.where(weekly.promo_flag == 1, np.maximum(weekly.units - weekly.naive_base_units, 0.0), 0.0).round(1)
    weekly["naive_base_units"] = weekly.naive_base_units.round(1)

    # trade costs: partial funding + merch fees + coupon redemption
    wk = weekly.set_index(["sku_id", "retailer", "t"]).sort_index()
    cols = {"trade_cost": [], "units_on_promo": [], "brand_funding_share": [], "everyday": [], "coupon_cost": []}
    keep = []
    for i, ev in cal.iterrows():
        try:
            g = wk.loc[(ev.sku_id, ev.retailer)].loc[ev.start_idx: ev.start_idx + ev.dur - 1]
        except KeyError:
            continue
        if len(g) < ev.dur:
            continue
        keep.append(i)
        promo_units = float(g.units.sum())
        everyday = float(g.everyday_price.iloc[0])
        share = float(RNG.uniform(0.34, 0.48))
        allowance = ev.depth * everyday * share * promo_units
        stores = float(g.stores_selling.mean())
        fees = ((6.0 * stores if ev.display else 0.0) + (3.9 * stores if ev.feature else 0.0)) * ev.dur
        coupon_cost = (COUPON_FACE * COUPON_REDEMPTION * promo_units) if ev.coupon else 0.0
        cols["trade_cost"].append(round(allowance + fees + coupon_cost, 2))
        cols["units_on_promo"].append(round(promo_units, 1))
        cols["brand_funding_share"].append(round(share, 2))
        cols["everyday"].append(everyday)
        cols["coupon_cost"].append(round(coupon_cost, 2))
    cal2 = cal.loc[keep].reset_index(drop=True)
    for k, v in cols.items():
        cal2[k] = v
    cal2["wholesale_price"] = (cal2.everyday * WHOLESALE_SHARE).round(3)
    return weekly, cal2


if __name__ == "__main__":
    weekly, events = simulate()
    weekly.to_csv("data/weekly.csv", index=False)
    events.to_csv("data/events.csv", index=False)
    gross_rev = (weekly.units * weekly.everyday_price * WHOLESALE_SHARE).sum()
    print(f"weekly rows: {len(weekly)}, events: {len(events)}")
    print(f"promo week share: {weekly.promo_flag.mean():.2%}")
    print(f"trade spend / gross wholesale revenue: {events.trade_cost.sum() / gross_rev:.1%}")
    print(f"coupon events: {int(events.coupon.sum())}, exec-fail: {int(events.exec_fail.sum())}")
