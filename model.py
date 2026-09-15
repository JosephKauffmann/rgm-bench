"""SCAN*PRO-family promotion-effectiveness engine for the Liftbench demo (v3).

Per SKU-retailer log-linear OLS (Van Heerde & Neslin, "Sales Promotion Models"; Wittink
et al.). The model separates the two elasticities RGM practice separates: BASE price
elasticity (how the baseline responds to list-price changes, identified off staggered
list increases) and PROMO price elasticity (uplift as a function of discount depth),
plus mechanics (display, feature, coupon), pre/post-promo dips, and cannibalization.
Baseline = fitted model with all PROMOTIONAL activity neutralized; list-price effects
stay in the baseline, where they belong. Duan (1983) smearing retransformation.
Event metrics use standard Base/Lift/ROI conventions; ROI is NET (breakeven 0%).
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

WHOLESALE_SHARE = 0.66

weekly = pd.read_csv("data/weekly.csv", parse_dates=["week_end"])
events = pd.read_csv("data/events.csv", parse_dates=["start_week", "end_week"])

HOLIDAY_COLS = ["newyear", "thanksgiving", "christmas", "july4", "memorial", "backtoschool"]
HOLIDAY_DATES = {
    "newyear": ["2024-01-06", "2024-01-13", "2025-01-04", "2025-01-11", "2026-01-03", "2026-01-10"],
    "thanksgiving": ["2023-11-25", "2024-11-30", "2025-11-29"],
    "christmas": ["2023-12-23", "2023-12-30", "2024-12-21", "2024-12-28", "2025-12-20", "2025-12-27"],
    "july4": ["2024-07-06", "2025-07-05", "2026-07-04"],
    "memorial": ["2024-05-25", "2025-05-31", "2026-05-30"],
    "backtoschool": ["2023-09-16", "2024-08-24", "2024-08-31", "2025-08-23", "2025-08-30", "2026-08-22", "2026-08-29"],
}
PRE_BUILD_COLS = ["pre_thanksgiving", "pre_christmas", "pre_july4", "pre_memorial"]
for h, dates in HOLIDAY_DATES.items():
    weekly[h] = weekly.week_end.isin(pd.to_datetime(dates)).astype(float)
for h in ("thanksgiving", "christmas", "july4", "memorial"):
    pre_dates = [pd.Timestamp(d) - pd.Timedelta(weeks=1) for d in HOLIDAY_DATES[h]]
    weekly[f"pre_{h}"] = weekly.week_end.isin(pre_dates).astype(float)

woy = weekly.week_end.dt.isocalendar().week.astype(float)
weekly["sin1"], weekly["cos1"] = np.sin(2 * np.pi * woy / 52), np.cos(2 * np.pi * woy / 52)
weekly["sin2"], weekly["cos2"] = np.sin(4 * np.pi * woy / 52), np.cos(4 * np.pi * woy / 52)
weekly["ln_units"] = np.log(weekly.units)
weekly["ln_price_ratio"] = np.log(weekly.price / weekly.everyday_price)   # promo depth term
weekly["ln_stores"] = np.log(weekly.stores_selling)
weekly["ln_list_rel"] = np.log(weekly.everyday_price
                               / weekly.groupby(["sku_id", "retailer"]).everyday_price.transform("first"))
weekly["sib_np"] = (weekly.sibling_promo * (1 - weekly.promo_flag)).astype(float)

FEATURES = ["const", "t_n", "ln_stores", "sin1", "cos1", "sin2", "cos2",
            *HOLIDAY_COLS, *PRE_BUILD_COLS,
            "ln_list_rel",                                    # BASE price elasticity (stays in baseline)
            "ln_price_ratio", "display", "feature", "coupon",  # promo response + mechanics
            "sib_np", "pre_promo", "post_promo"]
PROMO_COLS = ("ln_price_ratio", "display", "feature", "coupon", "sib_np", "pre_promo", "post_promo")

def design(d: pd.DataFrame) -> np.ndarray:
    cols = {"const": 1.0, "t_n": d.t / 104.0, "ln_stores": d.ln_stores,
            "sin1": d.sin1, "cos1": d.cos1, "sin2": d.sin2, "cos2": d.cos2,
            **{h: d[h] for h in HOLIDAY_COLS}, **{h: d[h] for h in PRE_BUILD_COLS},
            "ln_list_rel": d.ln_list_rel, "ln_price_ratio": d.ln_price_ratio,
            "display": d.display.astype(float), "feature": d.feature.astype(float),
            "coupon": d.coupon.astype(float), "sib_np": d.sib_np,
            "pre_promo": d.pre_promo.astype(float), "post_promo": d.post_promo.astype(float)}
    return pd.DataFrame(cols)[FEATURES].to_numpy()

IDX = {f: i for i, f in enumerate(FEATURES)}

# Stage 1: BASE price elasticity, pooled per SKU across banners with banner intercepts.
# A single banner's history can't separate a list increase from trend; the STAGGERED
# increase weeks across banners can. This is the identification the staggering exists for.
def pooled_base_pe(g_sku: pd.DataFrame, ridx: dict) -> tuple[float, float]:
    X0 = design(g_sku)
    dums = np.zeros((len(g_sku), len(ridx)))
    for j, r in enumerate(g_sku.retailer):
        dums[j, ridx[r]] = 1.0
    X = np.hstack([dums, X0[:, 1:]])          # banner intercepts replace the global const
    y = g_sku.ln_units.to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    se = np.sqrt(np.abs(np.diag(np.linalg.pinv(X.T @ X))) * (resid @ resid) / dof)
    i = len(ridx) - 1 + IDX["ln_list_rel"]    # offset: const dropped, dummies prepended
    return float(beta[i]), float(se[i])

BASE_PE, BASE_PE_SE = {}, {}
for sku, g_sku in weekly.groupby("sku_id"):
    ridx = {r: j for j, r in enumerate(sorted(g_sku.retailer.unique()))}
    BASE_PE[sku], BASE_PE_SE[sku] = pooled_base_pe(g_sku.sort_values(["retailer", "t"]), ridx)

# Stage 2: per SKU-retailer fits with the pooled base-PE term as a fixed offset.
# Two passes: fit, flag promo weeks whose response is far below the fitted response
# (promos that likely never executed on shelf), refit without them. Leaving them in
# attenuates every promo coefficient toward zero.
def fit_pair(g: pd.DataFrame, base_pe: float, keep: np.ndarray):
    X = design(g)
    off = base_pe * g.ln_list_rel.to_numpy()
    y = g.ln_units.to_numpy() - off
    Xf = np.delete(X, IDX["ln_list_rel"], axis=1)
    beta, *_ = np.linalg.lstsq(Xf[keep], y[keep], rcond=None)
    resid = y - Xf @ beta
    dof = max(keep.sum() - Xf.shape[1], 1)
    se = np.sqrt(np.abs(np.diag(np.linalg.pinv(Xf[keep].T @ Xf[keep]))) * (resid[keep] @ resid[keep]) / dof)
    calm_mask = (g.promo_flag == 0).to_numpy() & keep
    smear = float(np.exp(resid[calm_mask]).mean())
    sigma = float(resid[calm_mask].std())
    return X, resid, sigma, np.insert(beta, IDX["ln_list_rel"], base_pe), np.insert(se, IDX["ln_list_rel"], 0.0), smear

weekly["suspect_fail"] = False
weekly["suspect_stockout"] = False
for (sku, ret), g in weekly.groupby(["sku_id", "retailer"]):
    g = g.sort_values("t")
    _, resid, sigma, *_ = fit_pair(g, BASE_PE[sku], np.ones(len(g), bool))
    low = resid < -2.5 * sigma
    weekly.loc[g.index[(g.promo_flag == 1).to_numpy() & low], "suspect_fail"] = True
    # a calm week collapsing far below the fitted baseline is a distribution problem
    # (stockout, lost shelf), not demand: flag it and keep it out of the refit
    weekly.loc[g.index[(g.promo_flag == 0).to_numpy() & low], "suspect_stockout"] = True

BASE_PE, BASE_PE_SE = {}, {}
excl = weekly.suspect_fail | weekly.suspect_stockout
for sku, g_sku in weekly[~excl].groupby("sku_id"):
    ridx = {r: j for j, r in enumerate(sorted(g_sku.retailer.unique()))}
    BASE_PE[sku], BASE_PE_SE[sku] = pooled_base_pe(g_sku.sort_values(["retailer", "t"]), ridx)

fits = {}
weekly["model_base"] = np.nan
weekly["model_fit"] = np.nan
for (sku, ret), g in weekly.groupby(["sku_id", "retailer"]):
    g = g.sort_values("t")
    keep = (~(g.suspect_fail | g.suspect_stockout)).to_numpy()
    X, _, _, beta_full, se_full, smear = fit_pair(g, BASE_PE[sku], keep)
    se_full[IDX["ln_list_rel"]] = BASE_PE_SE[sku]
    Xb = X.copy()
    for c in PROMO_COLS:
        Xb[:, IDX[c]] = 0.0
    weekly.loc[g.index, "model_base"] = np.exp(Xb @ beta_full) * smear
    weekly.loc[g.index, "model_fit"] = np.exp(X @ beta_full) * smear
    fits[(sku, ret)] = dict(beta=beta_full.tolist(), se=se_full.tolist(), smear=smear)

sku_names = weekly.groupby("sku_id").sku_name.first()

# ---------------- event metrics + mechanics waterfall ----------------
# Waterfall splits use SKU-level response coefficients (the mean across that SKU's four
# pair fits): a single pair's noisy mechanic estimate can go slightly negative, and a
# negative display bar is an estimation artifact, not a story. The per-event deviation
# from the SKU-level response lands in the unexplained bar.
SKU_COEF = {}
for sku in weekly.sku_id.unique():
    betas = [np.array(fits[(s, r)]["beta"]) for (s, r) in fits if s == sku]
    mean_b = np.mean(betas, axis=0)
    SKU_COEF[sku] = {c: float(mean_b[IDX[c]]) for c in ("ln_price_ratio", "display", "feature", "coupon")}

wk = weekly.set_index(["sku_id", "retailer", "t"]).sort_index()
rows, waterfalls = [], []
for _, ev in events.iterrows():
    try:
        g = wk.loc[(ev.sku_id, ev.retailer)].loc[ev.start_idx: ev.start_idx + ev.dur - 1]
    except KeyError:
        continue
    actual = g.units.sum()
    wholesale = ev.everyday * WHOLESALE_SHARE
    margin = wholesale - g.cogs.iloc[0]
    f = fits[(ev.sku_id, ev.retailer)]
    beta = np.array(f["beta"])

    # dip costs from fitted coefficients (deal anticipation + pantry loading)
    dip_units = 0.0
    for off, coef in ((-1, "pre_promo"), (ev.dur, "post_promo")):
        try:
            r0 = wk.loc[(ev.sku_id, ev.retailer, ev.start_idx + off)]
            dip_units += float(r0.model_base) * (1 - np.exp(beta[IDX[coef]]))
        except KeyError:
            pass
    sib_cost_gp = 0.0
    for (s2, r2), f2 in fits.items():
        if r2 != ev.retailer or s2 == ev.sku_id:
            continue
        gg = weekly[(weekly.sku_id == s2) & (weekly.retailer == ev.retailer)
                    & (weekly.t.between(ev.start_idx, ev.start_idx + ev.dur - 1)) & (weekly.promo_flag == 0)]
        if len(gg) and gg.line.iloc[0] == g.line.iloc[0]:
            b2 = np.array(f2["beta"])
            lost = gg.model_base.sum() * (1 - np.exp(b2[IDX["sib_np"]]))
            sib_cost_gp += lost * (gg.everyday_price.iloc[0] * WHOLESALE_SHARE - gg.cogs.iloc[0])

    for src, base_col in [("model", "model_base"), ("naive", "naive_base_units"), ("truth", "base_units_true")]:
        base = g[base_col].sum()
        lift = actual - base
        gp = lift * margin
        roi = (gp - ev.trade_cost) / ev.trade_cost
        rows.append(dict(event_id=ev.event_id, source=src, base_units=round(base, 1),
                         lift_units=round(lift, 1),
                         lift_pct=round(lift / base * 100, 1) if base > 0 else None,
                         gross_profit=round(gp, 2), net_roi_pct=round(roi * 100, 1),
                         gp_after_dips_cannibalization=round(gp - dip_units * margin - sib_cost_gp, 2) if src == "model" else None))

    # mechanics waterfall: the model's EXPECTED lift split into depth/display/feature/coupon
    # via log-space shares, plus an explicit execution/unexplained bar reconciling to the
    # realized total. Scaling components by realized lift would paint negative display bars
    # on under-executing events, which is not what a display does.
    base_m = g.model_base.sum()
    sc = SKU_COEF[ev.sku_id]
    # a mechanic whose fitted effect is not positive (rare-mechanic noise, e.g. the launch
    # SKU's sparse coupon history) contributes no bar; its lift lands in unexplained
    terms = dict(
        depth=max(sc["ln_price_ratio"] * float(np.log(1 - ev.depth)), 0.0) if ev.depth > 0 else 0.0,
        display=max(sc["display"], 0.0) * float(ev.display),
        feature=max(sc["feature"], 0.0) * float(ev.feature),
        coupon=max(sc["coupon"], 0.0) * float(ev.coupon),
    )
    tot = sum(terms.values())
    lift_m = actual - base_m
    exp_lift = base_m * (np.exp(tot) - 1.0)
    shares = {k: (v / tot if tot != 0 else 0.0) for k, v in terms.items()}
    comp = {f"{k}_units": round(shares[k] * exp_lift, 0) for k in terms}
    waterfalls.append(dict(event_id=ev.event_id, total_lift_units=round(lift_m, 0),
                           unexplained_units=round(lift_m, 0) - sum(comp.values()), **comp))
metrics = pd.DataFrame(rows)
wf = pd.DataFrame(waterfalls)
ev_out = events.merge(metrics[metrics.source == "model"].drop(columns="source"), on="event_id")
ev_out = ev_out.merge(wf, on="event_id")
ev_out["discount_pct"] = (ev_out.depth * 100).round(0)
truth_m = metrics[metrics.source == "truth"].set_index("event_id")
naive_m = metrics[metrics.source == "naive"].set_index("event_id")
ev_out["true_roi_pct"] = ev_out.event_id.map(truth_m.net_roi_pct)
ev_out["naive_roi_pct"] = ev_out.event_id.map(naive_m.net_roi_pct)
ev_out["naive_lift_pct"] = ev_out.event_id.map(naive_m.lift_pct)

# flag events the model suspects never executed on shelf (majority of weeks flagged)
susp = []
for _, ev in ev_out.iterrows():
    gw = wk.loc[(ev.sku_id, ev.retailer)].loc[ev.start_idx: ev.start_idx + ev.dur - 1]
    susp.append(bool(gw.suspect_fail.mean() >= 0.5))
ev_out["suspected_fail"] = susp
detect = dict(
    planted_failures=int(ev_out.exec_fail.sum()),
    caught=int((ev_out.exec_fail & ev_out.suspected_fail).sum()),
    false_alarms=int((~ev_out.exec_fail & ev_out.suspected_fail).sum()),
)

# ---------------- the two elasticities + mechanics, recovered vs true ----------------
def coef_table(col, true_col, mult=False):
    out = []
    for (sku, ret), f in fits.items():
        i = IDX[col]
        est, se = f["beta"][i], f["se"][i]
        tv = float(weekly[(weekly.sku_id == sku) & (weekly.retailer == ret)][true_col].iloc[0])
        est_v = float(np.exp(est)) if mult else est
        out.append(dict(sku_id=sku, retailer=ret, est=round(est_v, 2), se=round(se, 3), true=round(tv, 2)))
    return pd.DataFrame(out)

promo_pe = coef_table("ln_price_ratio", "true_promo_pe")
mech = {m: coef_table(m, f"true_{m}_mult", mult=True) for m in ("display", "feature", "coupon")}
base_pe = pd.DataFrame([dict(sku_id=sku, est=round(BASE_PE[sku], 2), se=round(BASE_PE_SE[sku], 3),
                             true=round(float(weekly[weekly.sku_id == sku].true_base_pe.iloc[0]), 2),
                             sku_name=sku_names[sku]) for sku in sorted(BASE_PE)])

def by_sku(df):
    t = df.groupby("sku_id").agg(est_mean=("est", "mean"), spread=("est", "std"), true=("true", "first")).round(2).reset_index()
    t["sku_name"] = t.sku_id.map(sku_names)
    return t

# precomputed curve points so the front-end only draws, never computes
DEPTH_GRID = [round(d * 0.02, 2) for d in range(0, 21)]          # 0..40% depth
LIST_GRID = [round(-0.10 + x * 0.01, 2) for x in range(0, 21)]   # -10%..+10% list change
pp_sku = by_sku(promo_pe).set_index("sku_id")
curves = []
for sku in sorted(BASE_PE):
    bpe, bse = BASE_PE[sku], BASE_PE_SE[sku]
    ppe, ptrue = float(pp_sku.loc[sku, "est_mean"]), float(pp_sku.loc[sku, "true"])
    pse = float(pp_sku.loc[sku, "spread"]) / 2.0   # SE of the 4-pair mean
    btrue = float(weekly[weekly.sku_id == sku].true_base_pe.iloc[0])
    curves.append(dict(
        sku_id=sku, sku_name=sku_names[sku],
        promo_curve=[dict(depth_pct=int(d * 100),
                          uplift_pct_est=round(((1 - d) ** ppe - 1) * 100, 1),
                          uplift_pct_lo=round(min(((1 - d) ** (ppe - 1.96 * pse) - 1) * 100,
                                                  ((1 - d) ** (ppe + 1.96 * pse) - 1) * 100), 1),
                          uplift_pct_hi=round(max(((1 - d) ** (ppe - 1.96 * pse) - 1) * 100,
                                                  ((1 - d) ** (ppe + 1.96 * pse) - 1) * 100), 1),
                          uplift_pct_true=round(((1 - d) ** ptrue - 1) * 100, 1)) for d in DEPTH_GRID],
        base_curve=[dict(list_change_pct=int(x * 100),
                         baseline_change_pct_est=round(((1 + x) ** bpe - 1) * 100, 1),
                         baseline_change_pct_lo=round(min(((1 + x) ** (bpe - 1.96 * bse) - 1) * 100,
                                                         ((1 + x) ** (bpe + 1.96 * bse) - 1) * 100), 1),
                         baseline_change_pct_hi=round(max(((1 + x) ** (bpe - 1.96 * bse) - 1) * 100,
                                                         ((1 + x) ** (bpe + 1.96 * bse) - 1) * 100), 1),
                         baseline_change_pct_true=round(((1 + x) ** btrue - 1) * 100, 1)) for x in LIST_GRID],
    ))

elasticities = dict(
    curves=curves,
    base_pe=dict(by_sku=base_pe.to_dict(orient="records"),
                 note="Baseline response to LIST price changes. Pooled per SKU across banners "
                      "with banner intercepts; the staggered increase timing across banners is "
                      "what lets a price step be separated from trend at all, and pooling cuts "
                      "the standard error roughly in half. This is still the weakest-identified "
                      "number here: three list steps is thin evidence, the bands say so, and "
                      "the cross-SKU ordering should not be over-read."),
    promo_pe=dict(by_pair=promo_pe.to_dict(orient="records"), by_sku=by_sku(promo_pe).to_dict(orient="records"),
                  note="Uplift response to promotional discount depth."),
    mechanics={m: by_sku(mech[m]).to_dict(orient="records") for m in mech},
)

# dips + cannibalization (recovered vs simulated truth)
pre_c = [f["beta"][IDX["pre_promo"]] for f in fits.values()]
post_c = [f["beta"][IDX["post_promo"]] for f in fits.values()]
cann_c = [f["beta"][IDX["sib_np"]] for f in fits.values()]
effects = dict(
    pre_promo_dip_pct=round(float((np.exp(np.mean(pre_c)) - 1) * 100), 1), true_pre_promo_dip_pct=-4.0,
    post_promo_dip_pct=round(float((np.exp(np.mean(post_c)) - 1) * 100), 1), true_post_promo_dip_pct=-7.0,
    cannibalization_pct=round(float((np.exp(np.mean(cann_c)) - 1) * 100), 1), true_cannibalization_pct=-12.0,
    caption="Deal anticipation (week before), pantry loading (week after), and same-line sibling "
            "cannibalization: all fitted effects, recovered against known simulated values.",
)

# ---------------- baseline decomposition (stacked, sequential-additive) ----------------
def decompose(sku, ret):
    g = weekly[(weekly.sku_id == sku) & (weekly.retailer == ret)].sort_values("t")
    f = fits[(sku, ret)]
    beta, smear = np.array(f["beta"]), f["smear"]
    X = design(g)
    Xz = X.copy()
    for c in PROMO_COLS:
        Xz[:, IDX[c]] = 0.0
    def part(mask_cols):
        Xp = Xz.copy()
        for c in mask_cols:
            Xp[:, IDX[c]] = 0.0
        return np.exp(Xp @ beta) * smear
    seas_cols = ("sin1", "cos1", "sin2", "cos2")
    hol_cols = tuple(HOLIDAY_COLS + PRE_BUILD_COLS)
    core = part(seas_cols + hol_cols + ("ln_list_rel",))
    with_seas = part(hol_cols + ("ln_list_rel",))
    with_hol = part(("ln_list_rel",))
    full = np.exp(Xz @ beta) * smear
    return dict(sku_id=sku, retailer=ret, sku_name=sku_names[sku],
                weeks=g.week_end.dt.strftime("%Y-%m-%d").tolist(),
                units=g.units.round(1).tolist(),
                core=np.round(core, 1).tolist(),
                seasonality_add=np.round(with_seas - core, 1).tolist(),
                holiday_add=np.round(with_hol - with_seas, 1).tolist(),
                list_price_add=np.round(full - with_hol, 1).tolist(),
                baseline_total=np.round(full, 1).tolist(),
                promo_weeks=g.promo_flag.tolist())

decompositions = [decompose("BW-101", "Granite Peak Grocers"), decompose("BW-201", "ThriftLane")]

# ---------------- validation ----------------
promo_w = weekly[weekly.promo_flag == 1]
val = dict(
    baseline_mape_model=round(float((abs(weekly.model_base - weekly.base_units_true) / weekly.base_units_true).mean() * 100), 2),
    baseline_mape_naive=round(float((abs(weekly.naive_base_units - weekly.base_units_true) / weekly.base_units_true).mean() * 100), 2),
    promo_week_baseline_mape_model=round(float((abs(promo_w.model_base - promo_w.base_units_true) / promo_w.base_units_true).mean() * 100), 2),
    promo_week_baseline_mape_naive=round(float((abs(promo_w.naive_base_units - promo_w.base_units_true) / promo_w.base_units_true).mean() * 100), 2),
    promo_pe_median_abs_err=round(float((promo_pe.est - promo_pe.true).abs().median()), 3),
    base_pe_median_abs_err=round(float((base_pe.est - base_pe.true).abs().median()), 3),
    mechanics_mult_median_abs_err=round(float(pd.concat([mech[m].assign(err=(mech[m].est - mech[m].true).abs()) for m in mech]).err.median()), 3),
    fit_r2_ln_in_sample=round(float(1 - ((weekly.ln_units - np.log(weekly.model_fit)) ** 2).sum()
                                    / ((weekly.ln_units - weekly.ln_units.mean()) ** 2).sum()), 4),
    exec_fail_detection=detect,
    disclosure=("The synthetic world and the model are written by the same author, and the model is "
                "correctly specified for this world; read accuracy as a demonstration floor, not a "
                "production forecast. The comparator is a trailing average of recent calm weeks, an illustrative "
                "reference, not any vendor's method. The pooled base elasticity is plugged into the "
                "per-pair fits as a known value, so downstream baselines do not carry its uncertainty; "
                "the confidence band on the base-elasticity chart does."),
)
tm = metrics.pivot(index="event_id", columns="source", values="lift_units")
val["event_lift_mae_model"] = round(float((tm.model - tm.truth).abs().mean()), 1)
val["event_lift_mae_naive"] = round(float((tm.naive - tm.truth).abs().mean()), 1)
tr = metrics.pivot(index="event_id", columns="source", values="net_roi_pct")
val["events_misgraded_naive"] = int(((tr.naive > 0) != (tr.truth > 0)).sum())
val["events_misgraded_model"] = int(((tr.model > 0) != (tr.truth > 0)).sum())
gpv = metrics.pivot(index="event_id", columns="source", values="gross_profit")
val["gp_misread_naive"] = round(float((gpv.naive - gpv.truth).abs().sum()), 0)
val["gp_misread_model"] = round(float((gpv.model - gpv.truth).abs().sum()), 0)

TRAIN_END = 130
BASE_PE_H = {}
for sku, g_sku in weekly[weekly.t < TRAIN_END].groupby("sku_id"):
    ridx = {r: j for j, r in enumerate(sorted(g_sku.retailer.unique()))}
    BASE_PE_H[sku], _ = pooled_base_pe(g_sku.sort_values(["retailer", "t"]), ridx)
# One holdout, two jobs: grade the BASELINE out of sample (the measurement claim) and
# FORECAST total demand over the held-out 26 weeks with the planned promo calendar as a
# known input (the forecasting claim). Intervals from the train residual scale.
oos_m, oos_n = [], []
fc_pairs = {}
fc_err_all, fc_err_promo, fc_cov80 = [], [], []
fc_err_snaive, sn_brand = [], {}
for (sku, ret), g in weekly.groupby(["sku_id", "retailer"]):
    g = g.sort_values("t")
    tr_, te = g[g.t < TRAIN_END], g[g.t >= TRAIN_END]
    if len(tr_) < 60 or len(te) == 0:
        continue
    Xtr = np.delete(design(tr_), IDX["ln_list_rel"], axis=1)
    ytr = tr_.ln_units.to_numpy() - BASE_PE_H[sku] * tr_.ln_list_rel.to_numpy()
    beta_h, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    res = ytr - Xtr @ beta_h
    calm_h = res[(tr_.promo_flag == 0).to_numpy()]
    keep_h = ~(res < -2.5 * float(calm_h.std()))          # failed promos AND stockouts
    beta_h, *_ = np.linalg.lstsq(Xtr[keep_h], ytr[keep_h], rcond=None)
    res = ytr - Xtr @ beta_h
    smear_h = float(np.exp(res[(tr_.promo_flag == 0).to_numpy() & keep_h]).mean())
    sig_h = float(res[keep_h].std())
    Xte_full = design(te)
    off_te = BASE_PE_H[sku] * te.ln_list_rel.to_numpy()
    Xte_base = Xte_full.copy()
    for c in PROMO_COLS:
        Xte_base[:, IDX[c]] = 0.0
    base_oos = np.exp(np.delete(Xte_base, IDX["ln_list_rel"], axis=1) @ beta_h + off_te) * smear_h
    fc = np.exp(np.delete(Xte_full, IDX["ln_list_rel"], axis=1) @ beta_h + off_te) * smear_h
    m = (te.promo_flag == 1).to_numpy()
    if m.any():
        oos_m.extend((abs(base_oos[m] - te.base_units_true[m]) / te.base_units_true[m]).tolist())
        oos_n.extend((abs(te.naive_base_units[m] - te.base_units_true[m]) / te.base_units_true[m]).tolist())
    act = te.units.to_numpy()
    fc_err_all.extend((np.abs(fc - act) / act).tolist())
    if m.any():
        fc_err_promo.extend((np.abs(fc[m] - act[m]) / act[m]).tolist())
    # benchmark forecaster: same week last year
    g_idx = g.set_index("t").units
    sn = np.array([g_idx.get(t - 52, np.nan) for t in te.t])
    ok_sn = ~np.isnan(sn)
    fc_err_snaive.extend((np.abs(sn[ok_sn] - act[ok_sn]) / act[ok_sn]).tolist())
    sn_brand[(sku, ret)] = (te.t.to_numpy()[ok_sn], sn[ok_sn], act[ok_sn])
    lo80, hi80 = fc * np.exp(-1.282 * sig_h), fc * np.exp(1.282 * sig_h)
    fc_cov80.extend(((act >= lo80) & (act <= hi80)).tolist())
    fc_pairs[(sku, ret)] = dict(weeks=te.week_end.dt.strftime("%Y-%m-%d").tolist(),
                                actual=np.round(act, 1).tolist(), forecast=np.round(fc, 1).tolist(),
                                lo80=np.round(lo80, 1).tolist(), hi80=np.round(hi80, 1).tolist(),
                                lo95=np.round(fc * np.exp(-1.96 * sig_h), 1).tolist(),
                                hi95=np.round(fc * np.exp(1.96 * sig_h), 1).tolist(),
                                promo=te.promo_flag.tolist(), sig=sig_h)
val["oos_promo_week_baseline_mape_model"] = round(float(np.mean(oos_m) * 100), 2)
val["oos_promo_week_baseline_mape_naive"] = round(float(np.mean(oos_n) * 100), 2)
val["oos_note"] = "Model refit on weeks 1-130 only; both baselines graded on promo weeks 131-156, unseen by the model."

# brand rollup: sum the per-pair forecasts; interval half-widths add in quadrature
fkeys = sorted(fc_pairs)
fweeks = max((fc_pairs[k]["weeks"] for k in fkeys), key=len)
def roll(field):
    out = np.zeros(len(fweeks))
    for k in fkeys:
        d = fc_pairs[k]; n = len(d["weeks"])
        out[len(fweeks) - n:] += np.array(d[field])
    return out
br_act, br_fc = roll("actual"), roll("forecast")
hw80 = np.sqrt(sum((np.pad(np.array(fc_pairs[k]["hi80"]) - np.array(fc_pairs[k]["forecast"]),
                           (len(fweeks) - len(fc_pairs[k]["weeks"]), 0))) ** 2 for k in fkeys))
hw95 = np.sqrt(sum((np.pad(np.array(fc_pairs[k]["hi95"]) - np.array(fc_pairs[k]["forecast"]),
                           (len(fweeks) - len(fc_pairs[k]["weeks"]), 0))) ** 2 for k in fkeys))
val["forecast_mape_all_weeks"] = round(float(np.mean(fc_err_all) * 100), 2)
val["forecast_mape_promo_weeks"] = round(float(np.mean(fc_err_promo) * 100), 2)
val["forecast_interval80_coverage"] = round(float(np.mean(fc_cov80)), 2)
val["forecast_wape_brand"] = round(float(np.abs(br_fc - br_act).sum() / br_act.sum() * 100), 2)
val["forecast_mape_all_weeks_snaive"] = round(float(np.mean(fc_err_snaive) * 100), 2)
# same-week-last-year rolled up to brand, on the weeks where all pairs have a lag-52 value
sn_t = sorted(set.intersection(*[set(v[0]) for v in sn_brand.values()]))
sn_fc = sum(np.array([dict(zip(v[0], v[1]))[t] for t in sn_t]) for v in sn_brand.values())
sn_ac = sum(np.array([dict(zip(v[0], v[2]))[t] for t in sn_t]) for v in sn_brand.values())
val["forecast_wape_brand_snaive"] = round(float(np.abs(sn_fc - sn_ac).sum() / sn_ac.sum() * 100), 2)
val["forecast_note"] = ("26-week demand forecast from the week-130 refit, with the planned promo calendar "
                        "as a known input. Pair forecasts sum to the brand forecast; interval half-widths "
                        "add in quadrature, which assumes independent errors across pairs.")
show_fc = [("BW-101", "Granite Peak Grocers"), ("BW-201", "ThriftLane")]
forecast_out = dict(
    pairs=[dict(sku_id=s, retailer=r, sku_name=sku_names[s],
                **{k2: v for k2, v in fc_pairs[(s, r)].items() if k2 != "sig"}) for s, r in show_fc],
    brand=dict(weeks=fweeks, actual=np.round(br_act, 0).tolist(), forecast=np.round(br_fc, 0).tolist(),
               lo80=np.round(br_fc - hw80, 0).tolist(), hi80=np.round(br_fc + hw80, 0).tolist(),
               lo95=np.round(br_fc - hw95, 0).tolist(), hi95=np.round(br_fc + hw95, 0).tolist(),
               pairs_rolled=len(fkeys)),
)

# A fairer rule-based comparator: the trailing median with a week-of-year index on top.
# The index is each calm week's units relative to a centered rolling median of calm units
# (learned on train weeks only), aggregated by week-of-year, applied out of sample.
weekly["woy_i"] = woy.to_numpy()
oos_s = []
for (sku, ret), g in weekly.groupby(["sku_id", "retailer"]):
    g = g.sort_values("t")
    tr_, te = g[g.t < TRAIN_END], g[g.t >= TRAIN_END]
    if len(tr_) < 60 or not (te.promo_flag == 1).any():
        continue
    calm = tr_[tr_.promo_flag == 0]
    level = calm.units.rolling(9, center=True, min_periods=3).median()
    idx_map = (calm.units / level).groupby(calm.woy_i).median()
    adj = te.naive_base_units * te.woy_i.map(idx_map).fillna(1.0)
    m = (te.promo_flag == 1).to_numpy()
    oos_s.extend((abs(adj[m] - te.base_units_true[m]) / te.base_units_true[m]).tolist())
val["oos_promo_week_baseline_mape_seasonal_naive"] = round(float(np.mean(oos_s) * 100), 2)

# ---------------- what-if: depth x mechanics grid ----------------
# Showcase selection: never an event the model flagged as non-executing, always at least
# one mechanic, realized ROI inside the grid's range (so the realized dot sits on the
# chart), and among those the three where the fitted grid lands closest to what actually
# happened, with the portfolio-wide planning error reported alongside.
MECH_COMBOS = [("none", 0, 0, 0), ("feature", 0, 1, 0), ("feature+display", 1, 1, 0), ("feature+display+coupon", 1, 1, 1)]
BUNDLE = {(0, 1, 0): "feature", (1, 1, 0): "feature+display", (1, 1, 1): "feature+display+coupon"}

def event_pricing(ev_row):
    f = fits[(ev_row.sku_id, ev_row.retailer)]
    beta = np.array(f["beta"])
    g = wk.loc[(ev_row.sku_id, ev_row.retailer)].loc[ev_row.start_idx: ev_row.start_idx + ev_row.dur - 1]
    base = g.model_base.sum()
    margin = ev_row.everyday * WHOLESALE_SHARE - g.cogs.iloc[0]
    stores = float(g.stores_selling.mean())

    def cell(depth, D, F, C):
        mult = np.exp((beta[IDX["ln_price_ratio"]] * np.log(1 - depth) if depth > 0 else 0.0)
                      + beta[IDX["display"]] * D + beta[IDX["feature"]] * F + beta[IDX["coupon"]] * C)
        pred_units = base * mult
        lift = pred_units - base
        fees = ((6.0 * stores if D else 0.0) + (3.9 * stores if F else 0.0)) * ev_row.dur
        coup = 0.75 * 0.08 * pred_units if C else 0.0
        cost = depth * ev_row.everyday * ev_row.brand_funding_share * pred_units + fees + coup
        roi = float((lift * margin - cost) / cost * 100) if cost > 0 else None
        return float(lift), float(cost), roi
    return cell

def expected_grid(ev_row, bundle) -> dict:
    cell = event_pricing(ev_row)
    g0 = wk.loc[(ev_row.sku_id, ev_row.retailer)].loc[ev_row.start_idx: ev_row.start_idx + ev_row.dur - 1]
    margin0 = ev_row.everyday * WHOLESALE_SHARE - g0.cogs.iloc[0]
    # the fitted dip + cannibalization charge for this window is (near-)invariant to the
    # cell chosen, so every cell also gets the fuller-accounting ROI tab 3 argues for
    charge = float(ev_row.gross_profit - ev_row.gp_after_dips_cannibalization)
    cells = []
    for mech_name, D, F, C in MECH_COMBOS:
        for depth in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            lift, cost, roi = cell(depth, D, F, C)
            fuller = float((lift * margin0 - charge - cost) / cost * 100) if cost > 0 else None
            cells.append(dict(mechanics=mech_name, depth=int(depth * 100),
                              pred_lift_units=round(lift, 0), trade_cost=round(cost, 0),
                              net_roi_pct=round(roi, 1) if roi is not None else None,
                              fuller_net_roi_pct=round(fuller, 1) if fuller is not None else None))
    return dict(event_id=ev_row.event_id, sku_id=ev_row.sku_id, sku_name=sku_names[ev_row.sku_id],
                retailer=ev_row.retailer, actual_depth=int(ev_row.depth * 100), actual_mechanics=bundle,
                realized_net_roi_pct=float(ev_row.net_roi_pct),
                realized_fuller_net_roi_pct=round(float((ev_row.gp_after_dips_cannibalization - ev_row.trade_cost)
                                                        / ev_row.trade_cost * 100), 1),
                dip_cannibalization_charge=round(charge, 0), cells=cells,
                note="Expected outcomes from the fitted response model. See the planner summary "
                     "for the portfolio-wide planning error and how these three were chosen.")

# Presentation filters use only what the model can know (suspected_fail), never the
# planted truth flag. Eligibility: at least one mechanic in a bundle the grid can
# represent, and the realized point inside the grid's ROI range so the dot reads
# against the curves. The portfolio-wide planning error is reported alongside so the
# showcase choice (closest fits, one per SKU) cannot overstate accuracy.
ok = ev_out[~ev_out.suspected_fail & (ev_out.depth > 0)].copy()
cands, gaps = [], []
for _, r in ok.iterrows():
    key = (int(r.display), int(r.feature), int(r.coupon))
    if key not in BUNDLE:
        continue
    cell = event_pricing(r)
    _, _, pred_roi = cell(float(r.depth), *key)
    if pred_roi is None:
        continue
    gaps.append(abs(pred_roi - r.net_roi_pct))
    rois = [cell(d, D, F, C)[2] for _, D, F, C in MECH_COMBOS for d in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)]
    if not (min(rois) <= r.net_roi_pct <= max(rois)):
        continue
    cands.append((abs(pred_roi - r.net_roi_pct), r, BUNDLE[key]))
cands.sort(key=lambda x: x[0])
whatifs, used_sku = [], set()
for _, r, bundle in cands:
    if r.sku_id in used_sku:
        continue
    whatifs.append(expected_grid(r, bundle))
    used_sku.add(r.sku_id)
    if len(whatifs) == 3:
        break
whatif_summary = dict(
    eligible_events=len(gaps),
    median_abs_planning_error_pts=round(float(np.median(gaps)), 1),
    p90_abs_planning_error_pts=round(float(np.percentile(gaps, 90)), 1),
    selection_note="The three events shown are the ones where the fitted grid lands closest "
                   "to what actually ran, so the realized point sits inside the chart. The "
                   "figures above are the portfolio-wide planning error.",
)

# ---------------- data quality: stockout detection graded vs planted truth ----------------
calm = weekly[weekly.promo_flag == 0]
val["stockout_detection"] = dict(
    planted_weeks=int(calm.oos_week.sum()),
    caught=int((calm.oos_week.astype(bool) & calm.suspect_stockout).sum()),
    false_alarms=int((~calm.oos_week.astype(bool) & calm.suspect_stockout).sum()),
)

# ---------------- trade efficiency: GP returned per trade dollar, by banner x SKU ----------------
eff_rows = []
for (ret, sku), g in ev_out.groupby(["retailer", "sku_id"]):
    spend, gp = float(g.trade_cost.sum()), float(g.gross_profit.sum())
    eff_rows.append(dict(retailer=ret, sku_id=sku, sku_name=sku_names[sku], events=int(len(g)),
                         trade_spend=round(spend, 0), gross_profit=round(gp, 0),
                         net_roi_pct=round((gp - spend) / spend * 100, 1)))
by_ret = [dict(retailer=r, trade_spend=round(float(g.trade_cost.sum()), 0),
               net_roi_pct=round(float((g.gross_profit.sum() - g.trade_cost.sum()) / g.trade_cost.sum() * 100), 1))
          for r, g in ev_out.groupby("retailer")]
by_sku_eff = [dict(sku_id=s, sku_name=sku_names[s], trade_spend=round(float(g.trade_cost.sum()), 0),
                   net_roi_pct=round(float((g.gross_profit.sum() - g.trade_cost.sum()) / g.trade_cost.sum() * 100), 1))
              for s, g in ev_out.groupby("sku_id")]
trade_efficiency = dict(cells=eff_rows, by_retailer=by_ret, by_sku=by_sku_eff,
                        note="Event-window gross profit returned per trade dollar.")

# ---------------- price the product: list-price what-if from the pooled base elasticity ----------------
price_whatif = []
for sku in sorted(BASE_PE):
    gs = weekly[weekly.sku_id == sku]
    last = gs[gs.t == gs.t.max()]
    rr = gs[gs.t >= gs.t.max() - 25].groupby("retailer").model_base.mean()
    annual_units = float(rr.sum() * 52)
    p0 = float((last.set_index("retailer").everyday_price * rr).sum() / rr.sum())
    cogs = float(gs.cogs.iloc[0])
    pe, se = BASE_PE[sku], BASE_PE_SE[sku]
    gp0 = (WHOLESALE_SHARE * p0 - cogs) * annual_units
    pts = []
    for xi in range(-10, 11):
        x = xi / 100.0
        vols = sorted((1 + x) ** (pe + k * 1.96 * se) for k in (-1, 0, 1))
        gps = sorted(((WHOLESALE_SHARE * p0 * (1 + x) - cogs) * annual_units * v - gp0) for v in vols)
        pts.append(dict(list_change_pct=xi,
                        vol_pct_lo=round((vols[0] - 1) * 100, 1), vol_pct_est=round((vols[1] - 1) * 100, 1),
                        vol_pct_hi=round((vols[2] - 1) * 100, 1),
                        rev_pct_est=round(((1 + x) * vols[1] - 1) * 100, 1),
                        gp_delta_lo=round(gps[0], 0), gp_delta_est=round(gps[1], 0), gp_delta_hi=round(gps[2], 0)))
    price_whatif.append(dict(sku_id=sku, sku_name=sku_names[sku], current_avg_price=round(p0, 2),
                             annual_base_units=round(annual_units, 0), base_pe=round(pe, 2), base_pe_se=round(se, 3),
                             points=pts,
                             note="Wholesale annual gross-profit impact of a LIST price change at the current "
                                  "run rate, from the pooled base elasticity; the band carries its standard error. "
                                  "Everyday volume only; promo response is tab 5's job."))

# ---------------- KPIs ----------------
def kpi_row(r) -> dict:
    return dict(event_id=r.event_id, sku_id=r.sku_id, sku_name=sku_names[r.sku_id], retailer=r.retailer,
                start_week=r.start_week.strftime("%Y-%m-%d"), discount_pct=float(r.discount_pct),
                lift_pct=float(r.lift_pct), net_roi_pct=float(0.0 if r.net_roi_pct == 0 else r.net_roi_pct))

# "no lift" example: the flagged event whose lift is closest to zero, the "we paid and
# got nothing" story, not the minimum-lift outlier (which reads as volume destruction)
flagged = ev_out[ev_out.suspected_fail].copy()
no_lift = flagged.loc[flagged.lift_pct.abs().idxmin()] if len(flagged) else None
kpis = dict(
    top_event=kpi_row(ok.sort_values("net_roi_pct", ascending=False).iloc[0]),
    worst_paid_event=kpi_row(ok.sort_values("net_roi_pct").iloc[0]),
    no_lift_example=kpi_row(no_lift) if no_lift is not None else None,
    pct_events_unprofitable=round(float((ev_out.net_roi_pct < 0).mean() * 100), 1),
    total_trade_spend=round(float(ev_out.trade_cost.sum()), 0),
    blended_net_roi_pct=round(float((ev_out.gross_profit.sum() - ev_out.trade_cost.sum()) / ev_out.trade_cost.sum() * 100), 1),
    after_dips_and_cannibalization=dict(   # the fuller accounting a CFO will ask for
        blended_net_roi_pct=round(float((ev_out.gp_after_dips_cannibalization.sum() - ev_out.trade_cost.sum())
                                        / ev_out.trade_cost.sum() * 100), 1),
        pct_events_unprofitable=round(float(((ev_out.gp_after_dips_cannibalization - ev_out.trade_cost) < 0).mean() * 100), 1),
        note="Event-window ROI grades the promo week itself; this row also charges each "
             "event for its anticipation dip, pantry-loading dip, and sibling cannibalization.",
    ),
    events_measured=int(len(ev_out)),
    imported_split=dict(   # the same portfolio, graded by the naive/imported baseline (Tab 1)
        blended_net_roi_pct=round(float((naive_m.gross_profit.sum() - ev_out.trade_cost.sum()) / ev_out.trade_cost.sum() * 100), 1),
        pct_events_unprofitable=round(float((ev_out.naive_roi_pct < 0).mean() * 100), 1),
        events_measured=int(len(ev_out)),
        losing_events_called_profitable=int(((ev_out.true_roi_pct < 0) & (ev_out.naive_roi_pct >= 0)).sum()),
        losing_events_called_profitable_by_model=int(((ev_out.true_roi_pct < 0) & (ev_out.net_roi_pct >= 0)).sum()),
    ),
)

# ---------------- app_data.json ----------------
brand_weekly = weekly.groupby("week_end").agg(
    units=("units", "sum"), model_base=("model_base", "sum"), naive_base=("naive_base_units", "sum"),
    dollars=("dollars", "sum")).round(1).reset_index()
brand_weekly["week_end"] = brand_weekly.week_end.dt.strftime("%Y-%m-%d")

pair_series = []
for (sku, ret), g in weekly.groupby(["sku_id", "retailer"]):
    g = g.sort_values("t")
    pair_series.append(dict(
        sku_id=sku, retailer=ret, sku_name=g.sku_name.iloc[0], line=g.line.iloc[0],
        weeks=g.week_end.dt.strftime("%Y-%m-%d").tolist(),
        units=g.units.round(1).tolist(), model_base=g.model_base.round(1).tolist(),
        naive_base=g.naive_base_units.round(1).tolist(), true_base=g.base_units_true.round(1).tolist(),
        price=g.price.tolist(), everyday_price=g.everyday_price.tolist(), promo=g.promo_flag.tolist(),
    ))

events_json = ev_out.drop(columns=["start_idx"], errors="ignore").copy()
events_json["start_week"] = events_json.start_week.dt.strftime("%Y-%m-%d")
events_json["end_week"] = events_json.end_week.dt.strftime("%Y-%m-%d")
events_json["sku_name"] = events_json.sku_id.map(sku_names)

app = dict(
    meta=dict(
        app="RGM Bench", brand="Bramblewick Foods (synthetic)",
        weeks=int(weekly.t.nunique()), skus=int(weekly.sku_id.nunique()),
        retailers=int(weekly.retailer.nunique()), events=int(len(ev_out)),
        provenance="All data is synthetic, produced by a seeded generator written for this demo. "
                   "Methods are from the published marketing-science literature: SCAN*PRO-family "
                   "log-linear response models (Wittink et al.; Van Heerde & Neslin, 'Sales Promotion "
                   "Models', Handbook of Marketing Decision Models) with Duan (1983) retransformation. "
                   "Event metrics use standard Base/Lift/ROI conventions from trade-promotion practice. "
                   "No employer data, systems, code, or client information is used or referenced.",
    ),
    kpis=kpis,
    brand_weekly=brand_weekly.to_dict(orient="records"),
    events=events_json.to_dict(orient="records"),
    event_source_comparison=metrics.to_dict(orient="records"),
    elasticities=elasticities,
    effects=effects,
    baseline_decompositions=decompositions,
    pair_series=pair_series,
    whatif=whatifs,
    whatif_summary=whatif_summary,
    forecast=forecast_out,
    trade_efficiency=trade_efficiency,
    price_whatif=price_whatif,
    validation=val,
)
# JavaScript's JSON.parse rejects literal NaN/Infinity; scrub non-finite floats to null
def scrub(o):
    if isinstance(o, dict):
        return {k: scrub(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [scrub(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o

app = scrub(app)
with open("data/app_data.json", "w") as f:
    json.dump(app, f, allow_nan=False)

print(json.dumps({k: v for k, v in val.items() if not isinstance(v, str)}, indent=2))
print(json.dumps({k: v for k, v in kpis.items() if isinstance(v, (int, float))}, indent=2))
print("effects:", json.dumps({k: v for k, v in effects.items() if isinstance(v, (int, float))}))
print("events:", len(ev_out), "| json MB:", round(len(json.dumps(app)) / 1e6, 2))
