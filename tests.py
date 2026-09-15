"""Sanity + integrity tests for the RGM Bench demo (v3 conventions)."""
import json
import re
import sys
import numpy as np
import pandas as pd

failures = []
n_checks = 0
def check(name, cond, detail=""):
    global n_checks
    n_checks += 1
    print(f"{'PASS' if cond else 'FAIL'}  {name} {detail}")
    if not cond:
        failures.append(name)

weekly = pd.read_csv("data/weekly.csv", parse_dates=["week_end"])
events = pd.read_csv("data/events.csv")
raw = open("data/app_data.json").read()
try:  # JavaScript's JSON.parse rejects NaN/Infinity; the app must load in a browser
    app = json.loads(raw, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    strict_ok = True
except ValueError:
    app = json.loads(raw)
    strict_ok = False
val, kpis = app["validation"], app["kpis"]
evdf = pd.DataFrame(app["events"])
check("app_data.json is strict JSON (no NaN/Infinity)", strict_ok)

# --- data integrity ---
check("no NaNs in weekly", not weekly.isna().any().any())
check("no nonpositive units", (weekly.units > 0).all())
check("promo share 12-25%", 0.12 <= weekly.promo_flag.mean() <= 0.25, f"({weekly.promo_flag.mean():.2%})")
p = weekly[weekly.promo_flag == 1]
check("promo price = everyday x (1-depth)", np.allclose(p.price, (p.everyday_price * (1 - p.depth)).round(2), atol=0.011))
check("ARP sits between promo price and everyday", ((p.arp >= p.price - 0.01) & (p.arp <= p.everyday_price + 0.01)).all())
check("charm price endings", weekly.everyday_price.apply(lambda x: round(x % 1, 2) in (0.29, 0.49, 0.79, 0.99)).all())
check("staggered list increases (>=2 price levels per pair)", (weekly.groupby(["sku_id", "retailer"]).everyday_price.nunique() >= 2).all())
inc_weeks = weekly[weekly.everyday_price.ne(weekly.groupby(["sku_id", "retailer"]).everyday_price.shift())
                   & weekly.t.gt(0)].groupby("retailer").t.unique()
check("increase timing differs across banners", len({tuple(sorted(v)) for v in inc_weeks}) >= 3)
check("distribution grows", (weekly.groupby(["sku_id", "retailer"]).stores_selling.last().to_numpy()
                             > weekly.groupby(["sku_id", "retailer"]).stores_selling.first().to_numpy()).mean() > 0.9)
check("launch pair has shorter history", weekly.groupby(["sku_id", "retailer"]).size().min() < 156)
check("coupon events exist and ride ad/hero promos", (events.coupon.sum() >= 30) and (events[events.coupon == 1].depth.min() >= 0.10),
      f"({int(events.coupon.sum())} coupon events)")
check("pre-promo dip weeks flagged in weekly", weekly.pre_promo.sum() > 100, f"({int(weekly.pre_promo.sum())})")

hol_dates = [d for ds in {
    "newyear": ["2024-01-06","2024-01-13","2025-01-04","2025-01-11","2026-01-03","2026-01-10"],
    "thanksgiving": ["2023-11-25","2024-11-30","2025-11-29"],
    "christmas": ["2023-12-23","2023-12-30","2024-12-21","2024-12-28","2025-12-20","2025-12-27"],
    "july4": ["2024-07-06","2025-07-05","2026-07-04"],
    "memorial": ["2024-05-25","2025-05-31","2026-05-30"],
    "backtoschool": ["2023-09-16","2024-08-24","2024-08-31","2025-08-23","2025-08-30","2026-08-22","2026-08-29"],
}.values() for d in ds]
hm = weekly.week_end.isin(pd.to_datetime(hol_dates))
check("holiday promo rate >= 1.6x normal", weekly[hm].promo_flag.mean() / weekly[~hm].promo_flag.mean() >= 1.6,
      f"({weekly[hm].promo_flag.mean():.1%} vs {weekly[~hm].promo_flag.mean():.1%})")
pre_hol = [pd.Timestamp(d) - pd.Timedelta(weeks=1) for d in hol_dates]
ph = weekly[weekly.week_end.isin(pre_hol) & (weekly.promo_flag == 0)]
base_calm = weekly[~weekly.week_end.isin(pre_hol) & ~hm & (weekly.promo_flag == 0)]
check("pre-holiday build visible in true base",
      ph.base_units_true.mean() / base_calm.base_units_true.mean() > 1.02,
      f"(ratio {ph.base_units_true.mean() / base_calm.base_units_true.mean():.3f})")
gross_rev = (weekly.units * weekly.everyday_price * 0.66).sum()
check("trade spend 8-30% of gross wholesale revenue", 0.08 <= events.trade_cost.sum() / gross_rev <= 0.30,
      f"({events.trade_cost.sum() / gross_rev:.1%})")
g = weekly[weekly.line == "Granola"].groupby(weekly.week_end.dt.month).base_units_true.mean()
b = weekly[weekly.line == "Bars"].groupby(weekly.week_end.dt.month).base_units_true.mean()
check("granola peak in Jan (resolution season)", g.idxmax() == 1, f"(month {g.idxmax()})")
check("bars peak in summer", b.idxmax() in (6, 7, 8), f"(month {b.idxmax()})")
check("base PE and promo PE differ in the world", (weekly.true_promo_pe.abs() > weekly.true_base_pe.abs()).all())

# --- model quality ---
check("model baseline MAPE < 6%", val["baseline_mape_model"] < 6, f"({val['baseline_mape_model']}%)")
check("OOS: model beats naive >1.5x on promo weeks",
      val["oos_promo_week_baseline_mape_naive"] / max(val["oos_promo_week_baseline_mape_model"], 1e-9) > 1.5,
      f"({val['oos_promo_week_baseline_mape_naive']} vs {val['oos_promo_week_baseline_mape_model']})")
check("OOS: model beats the seasonally adjusted rule >1.5x too",
      val["oos_promo_week_baseline_mape_seasonal_naive"] / max(val["oos_promo_week_baseline_mape_model"], 1e-9) > 1.5,
      f"({val['oos_promo_week_baseline_mape_seasonal_naive']} vs {val['oos_promo_week_baseline_mape_model']})")
check("seasonal adjustment actually applied (differs from naive)",
      val["oos_promo_week_baseline_mape_seasonal_naive"] != val["oos_promo_week_baseline_mape_naive"],
      f"({val['oos_promo_week_baseline_mape_seasonal_naive']} vs {val['oos_promo_week_baseline_mape_naive']})")
check("promo PE median abs err <= 0.35", val["promo_pe_median_abs_err"] <= 0.35, f"({val['promo_pe_median_abs_err']})")
check("base PE median abs err <= 0.35", val["base_pe_median_abs_err"] <= 0.35, f"({val['base_pe_median_abs_err']})")
check("mechanics multiplier median abs err <= 0.2", val["mechanics_mult_median_abs_err"] <= 0.2,
      f"({val['mechanics_mult_median_abs_err']})")
check("event lift MAE: model < naive", val["event_lift_mae_model"] < val["event_lift_mae_naive"])
check("misgraded events: model < naive", val["events_misgraded_model"] < val["events_misgraded_naive"],
      f"({val['events_misgraded_model']} vs {val['events_misgraded_naive']})")
det = val["exec_fail_detection"]
check("exec-fail detection catches >= half the planted failures", det["caught"] >= det["planted_failures"] / 2,
      f"({det['caught']}/{det['planted_failures']}, {det['false_alarms']} false alarms)")
check("exec-fail false alarms < 5% of clean events", det["false_alarms"] < 0.05 * (len(evdf) - det["planted_failures"]),
      f"({det['false_alarms']})")
check("disclosure present", "same author" in val["disclosure"])

# --- forecasting (the holdout worn as a forecast) ---
check("forecast beats naive-level errors at brand level", val["forecast_wape_brand"] < 6,
      f"({val['forecast_wape_brand']}%)")
check("weekly forecast MAPE sane (5-25%)", 5 <= val["forecast_mape_all_weeks"] <= 25,
      f"({val['forecast_mape_all_weeks']}%)")
fb = app["forecast"]["brand"]
check("brand forecast covers all 26 holdout weeks from all pairs",
      len(fb["weeks"]) == 26 and fb["pairs_rolled"] == weekly.groupby(["sku_id", "retailer"]).ngroups)
check("forecast intervals ordered lo95<lo80<fc<hi80<hi95",
      all(a < b < c < d < e for a, b, c, d, e in
          zip(fb["lo95"], fb["lo80"], fb["forecast"], fb["hi80"], fb["hi95"])))
check("interval coverage reported (honesty stat present)", 0 < val["forecast_interval80_coverage"] <= 1)
check("forecast beats same-week-last-year >1.5x (weekly)",
      val["forecast_mape_all_weeks_snaive"] / max(val["forecast_mape_all_weeks"], 1e-9) > 1.5,
      f"({val['forecast_mape_all_weeks_snaive']} vs {val['forecast_mape_all_weeks']})")
check("forecast beats same-week-last-year at brand level too",
      val["forecast_wape_brand"] < val["forecast_wape_brand_snaive"],
      f"({val['forecast_wape_brand']} vs {val['forecast_wape_brand_snaive']})")

# --- data quality: stockout detector graded vs planted truth ---
sd = val["stockout_detection"]
check("stockout detector catches >= 80% of planted stockout weeks",
      sd["caught"] >= 0.8 * sd["planted_weeks"], f"({sd['caught']}/{sd['planted_weeks']}, {sd['false_alarms']} false alarms)")
check("stockout false alarms < 1% of calm weeks",
      sd["false_alarms"] < 0.01 * (weekly.promo_flag == 0).sum(), f"({sd['false_alarms']})")

# --- trade efficiency ---
te_ = app["trade_efficiency"]
check("efficiency matrix covers all banner x SKU cells", len(te_["cells"]) == weekly.groupby(["sku_id", "retailer"]).ngroups)
check("efficiency cells sum to portfolio spend",
      abs(sum(c["trade_spend"] for c in te_["cells"]) - kpis["total_trade_spend"]) < 5)
check("efficiency spread is a real story (>25 ROI pts)",
      max(c["net_roi_pct"] for c in te_["cells"]) - min(c["net_roi_pct"] for c in te_["cells"]) > 25)

# --- price what-if ---
pw = app["price_whatif"]
check("price what-if covers all SKUs with 21-point curves", len(pw) == weekly.sku_id.nunique()
      and all(len(p["points"]) == 21 for p in pw))
p0pt = [q for q in pw[0]["points"] if q["list_change_pct"] == 0][0]
check("price what-if is zero at zero", p0pt["vol_pct_est"] == 0 and p0pt["gp_delta_est"] == 0)
check("price what-if bands ordered", all(q["gp_delta_lo"] <= q["gp_delta_est"] <= q["gp_delta_hi"]
      for p in pw for q in p["points"]))
check("volume falls when list price rises (est)", all(q["vol_pct_est"] < 0 for p in pw
      for q in p["points"] if q["list_change_pct"] > 0))

# --- the two elasticities + effects blocks ---
el = app["elasticities"]
bp = pd.DataFrame(el["base_pe"]["by_sku"])
pp = pd.DataFrame(el["promo_pe"]["by_sku"])
check("base PE table covers all SKUs with SEs", len(bp) == weekly.sku_id.nunique() and (bp.se > 0).all())
check("promo PE magnitudes exceed base PE (recovered)", (pp.est_mean.abs().mean() > bp.est.abs().mean()))
check("mechanics tables present (display/feature/coupon)", set(el["mechanics"]) == {"display", "feature", "coupon"})
check("display multiplier > feature multiplier (recovered)",
      pd.DataFrame(el["mechanics"]["display"]).est_mean.mean() > pd.DataFrame(el["mechanics"]["feature"]).est_mean.mean())
eff = app["effects"]
check("pre-promo dip recovered within 3pts", abs(eff["pre_promo_dip_pct"] - eff["true_pre_promo_dip_pct"]) <= 3,
      f"({eff['pre_promo_dip_pct']} vs {eff['true_pre_promo_dip_pct']})")
check("post-promo dip recovered within 3pts", abs(eff["post_promo_dip_pct"] - eff["true_post_promo_dip_pct"]) <= 3,
      f"({eff['post_promo_dip_pct']} vs {eff['true_post_promo_dip_pct']})")
check("cannibalization recovered within 3pts", abs(eff["cannibalization_pct"] - eff["true_cannibalization_pct"]) <= 3,
      f"({eff['cannibalization_pct']} vs {eff['true_cannibalization_pct']})")

# --- economics (net-ROI convention) ---
check("blended net ROI in +5..+30%", 5 <= kpis["blended_net_roi_pct"] <= 30, f"({kpis['blended_net_roi_pct']}%)")
check("unprofitable share 30-55%", 30 <= kpis["pct_events_unprofitable"] <= 55, f"({kpis['pct_events_unprofitable']}%)")
check("exec-fail events read deeply negative", evdf[evdf.exec_fail].net_roi_pct.median() < -60,
      f"({evdf[evdf.exec_fail].net_roi_pct.median()})")
check("KPI numerics are numbers", all(isinstance(kpis["top_event"][k], (int, float))
      for k in ("discount_pct", "lift_pct", "net_roi_pct")))
ad = kpis["after_dips_and_cannibalization"]
check("after-dips ROI below event-window ROI", ad["blended_net_roi_pct"] < kpis["blended_net_roi_pct"],
      f"({ad['blended_net_roi_pct']} vs {kpis['blended_net_roi_pct']})")
check("after-dips unprofitable share above event-window share",
      ad["pct_events_unprofitable"] > kpis["pct_events_unprofitable"])
check("no-lift example truly shows ~no lift", abs(kpis["no_lift_example"]["lift_pct"]) < 10,
      f"({kpis['no_lift_example']['lift_pct']}%)")
ws = app["whatif_summary"]
check("planner summary reports portfolio-wide error", ws["eligible_events"] > 100
      and 0 < ws["median_abs_planning_error_pts"] <= ws["p90_abs_planning_error_pts"])

# --- formula fidelity: recompute one event by hand ---
ev = evdf.iloc[3]
pair = [q for q in app["pair_series"] if q["sku_id"] == ev.sku_id and q["retailer"] == ev.retailer][0]
idx = [i for i, w in enumerate(pair["weeks"]) if ev.start_week <= w <= ev.end_week]
actual = sum(pair["units"][i] for i in idx)
base = sum(pair["model_base"][i] for i in idx)
lift = actual - base
sku_cogs = weekly[weekly.sku_id == ev.sku_id].cogs.iloc[0]
gp = lift * (ev.everyday * 0.66 - sku_cogs)
check("event lift matches hand computation", abs(lift - ev.lift_units) < 1.5, f"({lift:.1f} vs {ev.lift_units})")
check("gross profit matches", abs(gp - ev.gross_profit) / max(abs(ev.gross_profit), 1) < 0.02)
check("net ROI = (GP - cost)/cost", abs((ev.gross_profit - ev.trade_cost) / ev.trade_cost * 100 - ev.net_roi_pct) < 0.5)
check("lift% = lift/base", abs(ev.lift_units / ev.base_units * 100 - ev.lift_pct) < 0.5)

# --- mechanics waterfall: expected components + unexplained reconcile exactly ---
wsum = (evdf.depth_units + evdf.display_units + evdf.feature_units + evdf.coupon_units + evdf.unexplained_units)
check("waterfall components + unexplained equal total lift exactly", np.allclose(wsum, evdf.total_lift_units, atol=0.01))
promo_ev = evdf[(evdf.depth > 0) & ~evdf.exec_fail]
check("depth is the largest lift driver on average",
      promo_ev.depth_units.abs().mean() > promo_ev.display_units.abs().mean())
check("mechanics components are never negative", (evdf[["depth_units", "display_units", "feature_units", "coupon_units"]] >= 0).all().all())

# --- what-if: depth x mechanics grid ---
w = app["whatif"]
check("3 what-if events, 4 mechanics x 7 depths", len(w) == 3 and all(len(x["cells"]) == 28 for x in w))
check("no exec-fail event in what-if", all(not evdf[evdf.event_id == x["event_id"]].exec_fail.iloc[0] for x in w))
check("no suspected-fail event in what-if", all(not evdf[evdf.event_id == x["event_id"]].suspected_fail.iloc[0] for x in w))
check("every what-if event ran at least one mechanic", all(x["actual_mechanics"] != "none" for x in w))
for x in w:
    rois = [c["net_roi_pct"] for c in x["cells"] if c["net_roi_pct"] is not None]
    check(f"realized point inside grid range ({x['event_id']})", min(rois) <= x["realized_net_roi_pct"] <= max(rois),
          f"({x['realized_net_roi_pct']} in {min(rois):.0f}..{max(rois):.0f})")
check("what-if carries realized point + actual mechanics",
      all("realized_net_roi_pct" in x and "actual_mechanics" in x for x in w))
c0 = pd.DataFrame(w[0]["cells"])
check("richer mechanics -> more predicted lift at same depth",
      c0[(c0.depth == 20) & (c0.mechanics == "feature+display")].pred_lift_units.iloc[0]
      > c0[(c0.depth == 20) & (c0.mechanics == "none")].pred_lift_units.iloc[0])

# --- baseline decomposition (the stacked-area story) ---
dec = app["baseline_decompositions"]
check("2 baseline decompositions exported", len(dec) == 2)
d0 = dec[0]
recon = np.array(d0["core"]) + np.array(d0["seasonality_add"]) + np.array(d0["holiday_add"]) + np.array(d0["list_price_add"])
check("decomposition components sum to baseline", np.allclose(recon, np.array(d0["baseline_total"]), atol=1.5))
check("decomposition carries actuals + promo flags for overlay", len(d0["units"]) == len(d0["weeks"]) == len(d0["promo_weeks"]))

# --- naming / provenance (positive allowlist proves cleanliness without naming anything else) ---
ALLOWED_NAMES = {"Bramblewick Foods (synthetic)", "Fernhollow Market", "Granite Peak Grocers",
                 "ThriftLane", "Hollybrook Foods", "RGM Bench"}
found = {app["meta"]["brand"], app["meta"]["app"], *{q["retailer"] for q in app["pair_series"]}}
check("all proper names in allowlist", found <= ALLOWED_NAMES, f"(extra: {found - ALLOWED_NAMES})")
check("all SKU names carry the synthetic brand", all(q["sku_name"].startswith("Bramblewick") for q in app["pair_series"]))
check("provenance states synthetic + literature + no-employer",
      all(s in app["meta"]["provenance"] for s in ("synthetic", "literature", "No employer")))
check("json < 5MB", len(json.dumps(app)) < 5e6, f"({len(json.dumps(app))/1e6:.2f}MB)")

# --- front-end: index.html must carry the CURRENT data (run build_html.py after model.py) ---
html = open("index.html").read()
mtag = re.search(r'<script id="app-data" type="application/json">(.*?)</script>', html, re.S)
embedded = json.loads(mtag.group(1)) if mtag else None
check("index.html embeds current app_data.json", embedded == app)
check("index.html titled with the app name", re.search(r"<title>[^<]*RGM Bench", html) is not None)

if failures:
    print(f"\n{len(failures)} of {n_checks} checks FAILED: {failures}")
else:
    print(f"\n{n_checks} checks, all passed")
sys.exit(1 if failures else 0)
