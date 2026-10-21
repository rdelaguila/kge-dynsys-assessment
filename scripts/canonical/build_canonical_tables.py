"""Build the canonical manuscript tables from one canonical execution root.

Usage:
  python build_canonical_tables.py <execution_root> <manuscript.tex> <historical_validator.py> <out_dir> [--allow-pending]
  python build_canonical_tables.py --public <execution_root> <out_dir> [--allow-pending]

--public writes only csv/<label>.csv and TABLE_VALUES.json (regenerated values, sources, producer, bolding rule); it
needs neither the manuscript LaTeX nor the historical validator, and makes no comparison with printed values.

Writes to <out_dir>:
  TABLE_VALUES_CANONICAL.json, csv/<label>.csv, TABLES_CANONICAL.tex, TABLES_FOR_AUTHOR.md,
  TABLE_MAPPING.csv, NUMERIC_DELTA_REPORT.md
Old (printed) values come from the constants of the historical validator, parsed with ast
(the module is never imported). Captions, labels and prose are never edited: in
TABLES_CANONICAL.tex only the tabular body between \\midrule and \\bottomrule is replaced.
Bolding is mechanical and its rule is stated per table in TABLE_VALUES_CANONICAL.json.
"""
from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

PUBLIC = len(sys.argv) > 1 and sys.argv[1] == "--public"
if PUBLIC:
    ROOT, OUT = (Path(a).resolve() for a in sys.argv[2:4])
    TEX = VALIDATOR = None
else:
    ROOT, TEX, VALIDATOR, OUT = (Path(a).resolve() for a in sys.argv[1:5])
ALLOW_PENDING = "--allow-pending" in sys.argv
sys.path.insert(0, str(ROOT / "src"))

DS = ["FB15k-237", "codex-m", "WN18RR"]
NAME = {"FB15k-237": "FB15k-237", "codex-m": "CoDEx-Medium", "WN18RR": "WN18RR"}
LEVELS = (5.0, 10.0, 15.0)
N_LOGICAL = {"FB15k-237": (15505, 31011, 46517), "codex-m": (10310, 20620, 30930), "WN18RR": (4650, 9300, 13950)}
KGE4 = ("TransE", "DistMult", "MuRE", "ComplEx")
KGE3 = ("TransE", "DistMult", "MuRE")
POOL = {"FB15k-237": 310116, "codex-m": 206205, "WN18RR": 93003}

# ── historical (printed) values, parsed from the validator without importing it ──
HIST = {}
if PUBLIC:  # no printed values: every historical lookup is empty
    from collections import defaultdict
    HIST = defaultdict(dict)
for node in ([] if PUBLIC else ast.parse(VALIDATOR.read_text()).body):
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        name = node.targets[0].id
        if name in {"T4_KGE", "T4_A2", "T5", "T5_MEAN", "T6", "T7", "T8", "C2", "C3", "C4", "T9", "WS", "S_F1",
                    "S_NDCG", "S_STATS", "T10", "PREV", "SWAP", "T12"}:
            HIST[name] = eval(compile(ast.Expression(node.value), "<hist>", "eval"), {"__builtins__": {}})


class Pending(Exception):
    pass


def load(rel):
    p = ROOT / rel
    if not p.exists():
        raise Pending(rel)
    return json.loads(p.read_text())


def loadgz(rel):
    p = ROOT / rel
    if not p.exists():
        raise Pending(rel)
    return json.load(gzip.open(p, "rt"))


def r(x, nd):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def f(x, nd):
    return f"{r(x, nd):.{nd}f}"


def fs(x, nd):  # signed
    v = r(x, nd)
    return ("+" if v >= 0 else "-") + f"{abs(v):.{nd}f}"


def fp(x):  # p-values as printed in the manuscript (5 dp below 0.01, else 4)
    return f(x, 5) if x < 0.01 else f(x, 4)


def thousands(n):
    return f"{int(n):,}"


def bold(s, on, math=False):
    if not on:
        return s
    return f"\\mathbf{{{s}}}" if math else f"\\textbf{{{s}}}"


def bold_max(vals, nd, unique_only=False, positive_only=True):
    """Indices to bold: all entries equal (at printed precision) to the maximum."""
    rv = [r(v, nd) for v in vals]
    m = max(rv)
    idx = [i for i, v in enumerate(rv) if v == m]
    if positive_only and m <= 0:
        return set()
    if unique_only and len(idx) > 1:
        return set()
    return set(idx)


TABLES = {}  # label -> dict(rows=[...], tex_rows=[...], header=[...], sources=[...], bold_rule=str, old=..., status=...)


def table(label, **kw):
    TABLES[label] = kw


# ── Tables 5-8 ──────────────────────────────────────────────────────────────
def kge_f1(ds, fname, m):
    return load(f"paper_results/{ds}/metrics/{fname}")["metrics"][m]["f1"]


def a2_f1(ds, fname):
    return load(f"paper_results/{ds}/post_fix_rerun/metrics/{fname}")["ode_postfix"]["f1"]


def table5():
    rows, tex = [], []
    for ds in DS:
        meth = ["A2", *KGE4]
        cols = {}
        for m in meth:
            v10 = [None, None]
            for j, lvl in enumerate((10.0, 15.0)):
                fn = f"metrics_random_entity_swap_{lvl}.json"
                v10[j] = a2_f1(ds, fn) if m == "A2" else kge_f1(ds, fn, m)
            if m == "ComplEx":
                mean, sd, n = kge_f1(ds, "metrics_random_entity_swap_5.0.json", m), None, 1
            else:
                fn = "metrics_random_entity_swap_5.0.json"
                if m == "A2":
                    v = [load(f"paper_results/{ds}/bootstrap_postfix/seed_{s}/random/metrics/{fn}")["ode_postfix"]["f1"] for s in range(1, 11)]
                else:
                    v = [load(f"paper_results/{ds}/bootstrap/seed_{s}/random/metrics/{fn}")["metrics"][m]["f1"] for s in range(1, 11)]
                mean, sd, n = float(np.mean(v)), float(np.std(v, ddof=1)), len(v)
            cols[m] = (mean, sd, n, v10[0], v10[1])
        b5 = bold_max([cols[m][0] for m in meth], 3)
        b10 = bold_max([cols[m][3] for m in meth], 3)
        b15 = bold_max([cols[m][4] for m in meth], 3)
        for i, m in enumerate(meth):
            mean, sd, n, x10, x15 = cols[m]
            rows.append({"dataset": ds, "method": m, "f1_5_mean": mean, "f1_5_sd": sd, "n_seeds_5": n, "f1_10": x10, "f1_15": x15})
            c5 = f"{f(mean, 3)} \\pm {f(sd, 3)}" if sd is not None else f(mean, 3)
            tex.append(f"{(chr(92) + 'textbf{' + NAME[ds] + '}') if i == 0 else ''} & {m + '*' if m == 'ComplEx' else m} & "
                       f"${bold(c5, i in b5, math=True)}$ & {bold(f(x10, 3), i in b10)} & {bold(f(x15, 3), i in b15)} \\\\")
    old = {(ds, m): v for ds, d in HIST["T5"].items() for m, v in d.items()}
    oldm = {(ds, m): v for ds, d in HIST["T5_MEAN"].items() for m, v in d.items()}
    table("tab:Table5Random", rows=rows, tex_rows=tex,
          sources=["paper_results/<ds>/bootstrap_postfix/seed_<1..10>/random/metrics/metrics_random_entity_swap_5.0.json (A2)",
                   "paper_results/<ds>/bootstrap/seed_<1..10>/random/metrics/metrics_random_entity_swap_5.0.json (TransE/DistMult/MuRE)",
                   "paper_results/<ds>/metrics/metrics_random_entity_swap_<lvl>.json (KGE seed 42)",
                   "paper_results/<ds>/post_fix_rerun/metrics/metrics_random_entity_swap_<lvl>.json (A2 seed 42)"],
          producer="run_kge_random5_seeds.py, run_all.py --stages random_ablation, milestone_postfix_section4_rerun.py",
          bold_rule="per dataset and column: all entries equal to the maximum at 3 dp (5% column compares means)",
          old_cells={**{(ds, m, "f1_5_mean"): oldm[(ds, m)][0] for ds, m in oldm}, **{(ds, m, "f1_5_sd"): oldm[(ds, m)][1] for ds, m in oldm},
                     **{(ds, m, "f1_5_mean"): v[0] for (ds, m), v in old.items() if v[0] is not None},
                     **{(ds, m, "f1_10"): v[1] for (ds, m), v in old.items()}, **{(ds, m, "f1_15"): v[2] for (ds, m), v in old.items()}},
          nd=3)


def logical_table(label, vtype, hist, unique_only):
    rows, tex = [], []
    for ds in DS:
        meth = ["A2", *KGE4]
        vals = {}
        for m in meth:
            vals[m] = [(a2_f1(ds, f"metrics_logical_{vtype}_{n}.json") if m == "A2" else kge_f1(ds, f"metrics_logical_{vtype}_{n}.json", m))
                       for n in N_LOGICAL[ds]]
        bset = [bold_max([vals[m][j] for m in meth], 3, unique_only=unique_only) for j in range(3)]
        for i, m in enumerate(meth):
            rows.append({"dataset": ds, "method": m, "f1_5": vals[m][0], "f1_10": vals[m][1], "f1_15": vals[m][2]})
            cells = " & ".join(bold(f(vals[m][j], 3), i in bset[j]) for j in range(3))
            tex.append(f"{(chr(92) + 'textbf{' + NAME[ds] + '}') if i == 0 else ''} & {m + '*' if m == 'ComplEx' else m} & {cells} \\\\")
    table(label, rows=rows, tex_rows=tex,
          sources=[f"paper_results/<ds>/metrics/metrics_logical_{vtype}_<n>.json (KGE, Path A, seed 42)",
                   f"paper_results/<ds>/post_fix_rerun/metrics/metrics_logical_{vtype}_<n>.json (A2, seed 42)"],
          producer="run_all.py --stages logical_ablation, milestone_postfix_section4_rerun.py",
          bold_rule=("per dataset and column: maximum at 3 dp, bolded only where unique and > 0" if unique_only
                     else "per dataset and column: all entries equal to the maximum at 3 dp"),
          old_cells={(ds, m, k): v[j] for ds, d in hist.items() for m, v in d.items() for j, k in enumerate(("f1_5", "f1_10", "f1_15"))},
          nd=3)


# ── Table 4 (native coverage) and Table 12 (consensus) ─────────────────────
def table4():
    rows, tex = [], []
    for ds in DS:
        a2 = loadgz(f"canonical_persist/detections/{ds}/native/native_rk4_50__A2.json.gz")
        assert a2["n"] == POOL[ds], (ds, a2["n"])
        rows.append({"dataset": ds, "method": "A2", "n_detected": a2["n_detected"], "pct_pool": 100 * a2["n_detected"] / a2["n"], "threshold": a2["threshold"]})
        tex.append(f"\\textbf{{{NAME[ds]}}} (pool {thousands(POOL[ds])}) & A2 (ODE) & {thousands(a2['n_detected'])} & "
                   f"{f(100 * a2['n_detected'] / a2['n'], 1)}\\% & knee (${f(a2['threshold'], 3)}$) \\\\")
        for m in KGE4:
            d = load(f"paper_results/{ds}/{m}/results.json")["detections"]
            rows.append({"dataset": ds, "method": m, "n_detected": d["n_detected"], "pct_pool": 100 * d["n_detected"] / POOL[ds],
                         "threshold": d["otsu_threshold"]})
            tex.append(f" & {m + '*' if m == 'ComplEx' else m} & {thousands(d['n_detected'])} & {f(100 * d['n_detected'] / POOL[ds], 1)}\\% & "
                       f"${f(d['otsu_threshold'], 2)}$ \\\\")
    old = {}
    for ds, (n, pct, thr) in HIST["T4_A2"].items():
        old.update({(ds, "A2", "n_detected"): n, (ds, "A2", "pct_pool"): pct, (ds, "A2", "threshold"): thr})
    for ds, d in HIST["T4_KGE"].items():
        for m, (n, pct, thr) in d.items():
            old.update({(ds, m, "n_detected"): n, (ds, m, "pct_pool"): pct, (ds, m, "threshold"): thr})
    table("tab:Table4Native", rows=rows, tex_rows=tex,
          sources=["canonical_persist/detections/<ds>/native/native_rk4_50__A2.json.gz (A2, 50 RK4 steps)",
                   "paper_results/<ds>/<model>/results.json (KGE, Otsu)"],
          producer="audit_phase2_E_rerun.py (native A2), run_all.py --stages native (KGE)",
          bold_rule="none", old_cells=old, nd=None)


def table12():
    rows, tex = [], []
    d = load("progress/tripartite_postfix.json")
    by = {x["dataset"]: x for x in d["rows"]}
    tot = {"confirmed_n": 0, "fp_risk_n": 0, "fn_risk_n": 0, "universal_consensus_n": 0}
    for ds in DS:
        x = by[ds]
        rows.append({"dataset": ds, "A2": x["ode_n"], "union_KGE": x["union_kge_n"], "shared": x["confirmed_n"], "KGE_only": x["fp_risk_n"],
                     "A2_only": x["fn_risk_n"], "all_method": x["universal_consensus_n"]})
        for k in tot:
            tot[k] += x[k]
        tex.append(f"{NAME[ds]} & {thousands(x['ode_n'])} & {thousands(x['union_kge_n'])} & {thousands(x['confirmed_n'])} & "
                   f"{thousands(x['fp_risk_n'])} & {thousands(x['fn_risk_n'])} & {thousands(x['universal_consensus_n'])} \\\\")
    rows.append({"dataset": "TOTAL", "shared": tot["confirmed_n"], "KGE_only": tot["fp_risk_n"], "A2_only": tot["fn_risk_n"], "all_method": tot["universal_consensus_n"]})
    tex.append(f"\\textbf{{TOTAL}} & --- & --- & \\textbf{{{thousands(tot['confirmed_n'])}}} & \\textbf{{{thousands(tot['fp_risk_n'])}}} & "
               f"\\textbf{{{thousands(tot['fn_risk_n'])}}} & \\textbf{{{thousands(tot['universal_consensus_n'])}}} \\\\")
    old = {}
    for ds, v in HIST["T12"].items():
        for k, x in zip(("A2", "union_KGE", "shared", "KGE_only", "A2_only", "all_method"), v):
            old[(ds, k)] = x
    for k, i in (("shared", 2), ("KGE_only", 3), ("A2_only", 4), ("all_method", 5)):
        old[("TOTAL", k)] = sum(v[i] for v in HIST["T12"].values())
    table("tab:Table12Tripar", rows=rows, tex_rows=tex, sources=["progress/tripartite_postfix.json",
          "canonical_persist/detections/<ds>/consensus/native_rk4_50__<partition>.json.gz (full sets)"],
          producer="compute_tripartite_postfix.py (persisted native A2 set + paper_results/<ds>/<model>/detections.json)",
          bold_rule="TOTAL row (as printed)", old_cells=old, keycols=("dataset",), nd=None)


# ── Table 10 and effective prevalence ──────────────────────────────────────
PANEL_ORDER = [("WN18RR", "transitivity", 5), ("codex-m", "cardinality", 15), ("WN18RR", "asymmetry", 15),
               ("FB15k-237", "transitivity", 15), ("WN18RR", "cardinality", 15), ("codex-m", "asymmetry", 10),
               ("codex-m", "transitivity", 5), ("codex-m", "cardinality", 5), ("FB15k-237", "cardinality", 5)]
DETS = ("ODE-A1", "ODE-A2", "ODE-A3", "TransE", "DistMult", "MuRE")


def table10_and_prevalence():
    fp_ = load("progress/20260623/full_panel.json")
    cells = {(c["dataset"], c["vtype"], c["level_pct"]): c for c in fp_["cells"]}
    rows, tex = [], []
    for ds, vt, lvl in PANEL_ORDER:
        c = cells[(ds, vt, lvl)]
        vals = [c["detectors"][d]["ndcg_at_K"] for d in DETS]
        b = bold_max(vals, 3, unique_only=True)
        prev = c["prevalence"]
        rows.append({"cell": f"{ds}/{vt}@{lvl}", "prevalence": prev, **{d: v for d, v in zip(DETS, vals)}})
        cellname = f"{'codex-m' if ds == 'codex-m' else ds} / {vt}@{lvl}\\%"
        tex.append(f"{cellname} & {f(prev, 4) if prev < 0.01 else f(prev, 3)} & " + " & ".join(bold(f(v, 3), i in b) for i, v in enumerate(vals)) + " \\\\")
    old = {}
    for ds, vt, lvl, prev, vals in HIST["T10"]:
        old[(f"{ds}/{vt}@{lvl}", "prevalence")] = prev
        for d, v in zip(DETS, vals):
            old[(f"{ds}/{vt}@{lvl}", d)] = v
    table("tab:Table10Fromp", rows=rows, tex_rows=tex, sources=["progress/20260623/full_panel.json"],
          producer="compute_full_panel.py (A1/A2 npz from milestone, A3 npz from run_sde_per_cell, KGE from post_analysis_capture)",
          bold_rule="unique row maximum at 3 dp; not bolded when all zero", old_cells=old, keycols=("cell",), nd=3)
    prow, ptex = [], []
    gen = {"asymmetry": "Asymmetry", "transitivity": "Closure corruption", "cardinality": "Cardinality"}
    for ds in DS:
        for vt in ("asymmetry", "transitivity", "cardinality"):
            cs = [cells[(ds, vt, l)] for l in (5, 10, 15)]
            prow.append({"dataset": ds, "generator": vt, **{f"n_pos_{l}": c["n_pos"] for l, c in zip((5, 10, 15), cs)},
                         **{f"n_eval_{l}": c["n_eval"] for l, c in zip((5, 10, 15), cs)}})
            ptex.append(f"{NAME[ds]} & {gen[vt]} & " + " & ".join(f"{thousands(c['n_pos'])} ({f(100 * c['n_pos'] / c['n_eval'], 2)}\\%)" for c in cs) + " \\\\")
    pold = {}
    for (ds, vt), v in HIST["PREV"].items():
        for l, n in zip((5, 10, 15), v):
            pold[(ds, vt, f"n_pos_{l}")] = n
    table("tab:effective_prevalence_ranking", rows=prow, tex_rows=ptex, sources=["progress/20260623/full_panel.json (n_pos, n_eval)"],
          producer="compute_full_panel.py", bold_rule="none", old_cells=pold, keycols=("dataset", "generator"), nd=None)
    au = [d["auroc"] for c in fp_["cells"] for d in c["detectors"].values() if d.get("auroc") is not None]
    return {"auroc_populated": len(au), "auroc_min": min(au), "auroc_max": max(au)}


# ── Section 4.5 ─────────────────────────────────────────────────────────────
def weight_sensitivity():
    w = load("progress/milestone_b_weight_sensitivity_WN18RR_10k_corrected.json")["conditions"]
    grid = {c: {(g["w_E"], g["w_sigma"]): g for g in w[c]["grid"]} for c in w}
    rows, tex = [], []
    for wE in (2.5, 5.0, 10.0):
        for wV in (50, 100, 200):
            a, b = grid["random_5pct"][(wE, wV)], grid["logical_asym_10pct"][(wE, wV)]
            rows.append({"w_E": wE, "w_V": wV, "random_f1": a["f1"], "random_jaccard": a["jaccard_vs_nominal"],
                         "asym_f1": b["f1"], "asym_jaccard": b["jaccard_vs_nominal"]})
            tex.append(f"{wE:.1f} & {int(wV)} & {f(a['f1'], 4)} & {f(a['jaccard_vs_nominal'], 4)} & {f(b['f1'], 4)} & {f(b['jaccard_vs_nominal'], 4)} \\\\")
    old = {}
    for wE, wV, rf, rj, af, aj in HIST["WS"]:
        old.update({(wE, wV, "random_f1"): rf, (wE, wV, "random_jaccard"): rj, (wE, wV, "asym_f1"): af, (wE, wV, "asym_jaccard"): aj})
    table("tab:weight_sensitivity", rows=rows, tex_rows=tex, sources=["progress/milestone_b_weight_sensitivity_WN18RR_10k_corrected.json"],
          producer="src/experiments/run_b_weight_sensitivity.py", bold_rule="none", old_cells=old, keycols=("w_E", "w_V"), nd=4)


STAGES = ["S0_zero_integration", "S1_data_only", "S2_data_sym_trans", "S3_plus_antisymmetry", "S4_plus_checker_bonus"]


def progressive():
    base = "progress/r2_1_progressive_ablation"
    for label, key, hist in (("tab:progressive_f1", "f1", HIST["S_F1"]), ("tab:progressive_ndcg", "ndcg_at_k", HIST["S_NDCG"])):
        rows, tex = [], []
        for ds in DS:
            seeds = [load(f"{base}/{ds}_seed_{s}.json") for s in range(1, 11)]
            assert all(x.get("ok", True) for x in seeds)
            means = [float(np.mean([x["stages"][st][key] for x in seeds])) for st in STAGES]
            b = bold_max(means, 4)
            rows.append({"dataset": ds, **{st[:2]: v for st, v in zip(STAGES, means)}})
            tex.append(f"{NAME[ds]} & " + " & ".join(bold(f(v, 4), i in b) for i, v in enumerate(means)) + " \\\\")
        table(label, rows=rows, tex_rows=tex, sources=[f"{base}/<ds>_seed_<1..10>.json"],
              producer="src/experiments/run_s0_s4_progressive_ablation.py", bold_rule="row maximum at 4 dp",
              old_cells={(ds, f"S{i}"): v for ds, vals in hist.items() for i, v in enumerate(vals)}, keycols=("dataset",), nd=4)
    st = load(f"{base}/stats_summary.json")
    rows, tex = [], []
    prev_ds = None
    for x in st["rows"]:
        sig = x["p_holm_12"] < 0.05
        c = x["contrast"].replace("-", "--")
        rows.append({"dataset": x["dataset"], "contrast": x["contrast"], **{k: x[k] for k in ("mean", "sd", "ci95_lo", "ci95_hi", "rank_biserial", "p_raw", "p_holm_12")}})
        tex.append(f"{NAME[x['dataset']] if x['dataset'] != prev_ds else ''} & {c} & {bold(fs(x['mean'], 4), sig)} & {f(x['sd'], 4)} & "
                   f"[{fs(x['ci95_lo'], 4)},{fs(x['ci95_hi'], 4)}] & {fs(x['rank_biserial'], 3)} & {fp(x['p_raw'])} & {bold(f(x['p_holm_12'], 4), sig)} \\\\")
        prev_ds = x["dataset"]
    old = {}
    for (ds, k), v in HIST["S_STATS"].items():
        for nm, x in zip(("mean", "sd", "ci95_lo", "ci95_hi", "rank_biserial", "p_raw", "p_holm_12"), v):
            old[(ds, f"S{k}-S{k - 1}", nm)] = x
    table("tab:progressive_stats", rows=rows, tex_rows=tex, sources=[f"{base}/stats_summary.json (bootstrap seed {st['bootstrap_seed']})"],
          producer="scripts/canonical/compute_s0s4_2x2_stats.py", bold_rule="mean and p_Holm bolded where p_Holm,12 < 0.05",
          old_cells=old, keycols=("dataset", "contrast"), nd=4)
    x = load("progress/milestone_c2_dynamics_symbolic_2x2_WN18RR_stats.json")
    cm = x["cell_means"]
    rows = [{"lambda_asym": "0", "bonus_off": cm["f1_zero_bonus_off"], "bonus_on": cm["f1_zero_bonus_on"]},
            {"lambda_asym": "ref", "bonus_off": cm["f1_ref_bonus_off"], "bonus_on": cm["f1_ref_bonus_on"]}]
    tex = [f"$\\lambda_{{\\mathrm{{asym}}}}=0$ & {f(cm['f1_zero_bonus_off'], 4)} & {f(cm['f1_zero_bonus_on'], 4)} \\\\",
           f"$\\lambda_{{\\mathrm{{asym}}}}=\\mathrm{{ref}}$ & {f(cm['f1_ref_bonus_off'], 4)} & {f(cm['f1_ref_bonus_on'], 4)} \\\\"]
    table("tab:dynamics_symbolic_2x2", rows=rows, tex_rows=tex,
          sources=["progress/milestone_c2_dynamics_symbolic_2x2_WN18RR.json", "progress/milestone_c2_dynamics_symbolic_2x2_WN18RR_stats.json"],
          producer="src/experiments/run_c2_dynamics_symbolic_2x2.py + scripts/canonical/compute_s0s4_2x2_stats.py", bold_rule="none",
          old_cells={("0", "bonus_off"): .2196, ("0", "bonus_on"): .2527, ("ref", "bonus_off"): .1832, ("ref", "bonus_on"): .1902},
          keycols=("lambda_asym",), nd=4, extra={"simple_effects_holm4": x["simple_effects_holm4"], "difference_of_differences": x["difference_of_differences"]})


# ── Appendix A configuration tables ─────────────────────────────────────────
def appendix_a():
    import config
    from run_all import get_constraint_config, _load_hyperparam_cache  # noqa: F401  (exec-root code)
    cache = {(ds, m): json.loads((ROOT / ".cache/hyperparams" / f"{ds}_{m}.json").read_text())["best_params"] for ds in DS for m in KGE4}
    rows, tex = [], []
    for ds in DS:
        for i, m in enumerate(KGE3):
            c = cache[(ds, m)]
            rows.append({"dataset": ds, "model": m, "loss": "MarginRankingLoss", "embedding_dim": c["embedding_dim"], "learning_rate": c["learning_rate"],
                         "margin": c["margin"], "negative_sampling": c["negative_sampling"]})
            lr = c["learning_rate"]
            lr_tex = f"$10^{{{int(round(np.log10(lr)))}}}$" if abs(lr - 10 ** round(np.log10(lr))) < 1e-15 else f"${lr:g}$"
            tex.append(f"{NAME[ds] if i == 0 else ''} & {m} & {c['embedding_dim']} & {lr_tex} & {c['margin']:.1f} & {c['negative_sampling']} \\\\")
    table("tab:app_kge_training", rows=rows, tex_rows=tex, sources=[".cache/hyperparams/<ds>_<model>.json (corrected held-out tuner)", "src/config.py"],
          producer="src/hyperparameter_tuning.py::tune_kge_held_out via run_all.py --stages tune", bold_rule="none",
          old_cells={(ds, m, k): v for ds in DS for m in KGE3 for k, v in (("embedding_dim", 100), ("learning_rate", 1e-4), ("margin", 1.0), ("negative_sampling", 10))},
          keycols=("dataset", "model"), nd=None,
          restructured=("The printed table has one value per stage for all three models and datasets (100 / 1e-4 / margin 1 / 10). "
                        "The corrected-tuner configurations differ by dataset and model, so the body is replaced by a "
                        "dataset x model block (Dataset & Model & d & eta & margin & negatives); the recipe rows are unchanged: "
                        f"epochs {config.NATIVE_EPOCHS}/{config.ABLATION_EPOCHS}, batch {config.NATIVE_BATCH}/{config.ABLATION_BATCH}, "
                        "Adam, PyKEEN random seed 42, MarginRankingLoss, TransE/DistMult/MuRE on MPS."))
    rows, tex = [], []
    for ds in DS:
        c = cache[(ds, "ComplEx")]
        rows.append({"dataset": ds, "loss": "SoftplusLoss", "embedding_dim": c["embedding_dim"], "learning_rate": c["learning_rate"], "negative_sampling": c["negative_sampling"]})
        e = int(np.floor(np.log10(c["learning_rate"])))
        mant = c["learning_rate"] / 10 ** e
        lr_tex = f"$10^{{{e}}}$" if abs(mant - 1) < 1e-9 else f"${mant:g}\\times10^{{{e}}}$"
        tex.append(f"{NAME[ds]} & SoftplusLoss & {c['embedding_dim']} & {lr_tex} & {c['negative_sampling']} \\\\")
    table("tab:app_complex_training", rows=rows, tex_rows=tex, sources=[".cache/hyperparams/<ds>_ComplEx.json"],
          producer="corrected held-out tuner (SoftplusLoss passed explicitly since 2240c14)", bold_rule="none",
          old_cells={("FB15k-237", "embedding_dim"): 200, ("FB15k-237", "learning_rate"): 5e-3, ("FB15k-237", "negative_sampling"): 25,
                     ("codex-m", "embedding_dim"): 200, ("codex-m", "learning_rate"): 5e-3, ("codex-m", "negative_sampling"): 25,
                     ("WN18RR", "embedding_dim"): 100, ("WN18RR", "learning_rate"): 5e-3, ("WN18RR", "negative_sampling"): 50},
          keycols=("dataset",), nd=None)


# ── Appendix B/C ───────────────────────────────────────────────────────────
def operator_swap():
    x = load("progress/r02_swap_means_recomputed.json")
    rows, tex = [], []
    for det, name in (("ODE-A2", "ODE (A2)"), ("DistMult", "DistMult"), ("TransE", "TransE"), ("MuRE", "MuRE")):
        d = x["per_detector_deltas"][det]
        w, l_, t = sum(v > 0 for v in d), sum(v < 0 for v in d), sum(v == 0 for v in d)
        mu = x["per_detector_mean_delta"][det]
        rows.append({"detector": det, "swap_wins": w, "swap_losses": l_, "ties": t, "mean_delta_f1": mu})
        tex.append(f"{name} & {w} / {l_} / {t} & ${fs(mu, 3)}$ \\\\")
    old = {}
    for det, (w, l_, t, mu) in HIST["SWAP"].items():
        old.update({(det, "swap_wins"): w, (det, "swap_losses"): l_, (det, "ties"): t, (det, "mean_delta_f1"): mu})
    table("tab:app_operator_swap", rows=rows, tex_rows=tex, sources=["progress/parallel_operator_sweep.json", "progress/r02_swap_means_recomputed.json"],
          producer="compute_parallel_operator_sweep.py + compute_r02_swap_means.py", bold_rule="none", old_cells=old, keycols=("detector",), nd=3,
          extra={k: x[k] for k in ("n_comparisons", "canonical_wins", "swap_wins", "ties") if k in x})


def table9_and_stats():
    from statistical_tests import bootstrap_ci, compute_final_statistical_report
    rows, tex = [], []
    for ds in DS:
        h, s = load(f"progress/milestone_c_h_asym_{ds}.json"), load(f"progress/sde_b6_wilcoxon_{ds}.json")
        assert len(h["per_seed"]) == 10 and len(s["per_seed"]) == 10
        a1 = float(np.mean([p["f1_zero"] for p in h["per_seed"]])); a2 = float(np.mean([p["f1_ref"] for p in h["per_seed"]]))
        a3 = float(np.mean([p["f1_ref"] for p in s["per_seed"]]))
        rows.append({"dataset": ds, "A1": a1, "A2": a2, "A3": a3})
        tex.append(f"{NAME[ds]} & {f(a1, 3)} & {f(a2, 3)} & {f(a3, 3)} \\\\")
    table("tab:Table9Threea", rows=rows, tex_rows=tex,
          sources=["progress/milestone_c_h_asym_<ds>.json (A1/A2, 100 RK4 steps)", "progress/sde_b6_wilcoxon_<ds>.json (A3)"],
          producer="run_c_h_asym_deterministic.py, run_sde_b6_wilcoxon.py", bold_rule="none",
          old_cells={(ds, k): v for ds, vals in HIST["T9"].items() for k, v in zip(("A1", "A2", "A3"), vals)}, keycols=("dataset",), nd=3)
    rep = compute_final_statistical_report(ROOT, write_output=False)["rows"]
    rnd_rows, rtex = [], []
    for x in [x for x in rep if "RANDOM" in x["family_id"]]:
        rnd_rows.append({"dataset": x["dataset"], "kge": x["comparison"], **{k: x[k] for k in ("mean_diff", "sd_diff", "rank_biserial", "p_raw", "p_holm")}})
        rtex.append(f"{NAME[x['dataset']]} & {x['comparison']} & ${fs(x['mean_diff'], 6)}$ & ${f(x['sd_diff'], 6)}$ & ${fs(x['rank_biserial'], 3)}$ & "
                    f"${f(x['p_raw'], 6)}$ & ${f(x['p_holm'], 6)}$ \\\\")
    old = {}
    for (ds, m), (mu, sd) in HIST["C2"].items():
        old.update({(ds, m, "mean_diff"): mu, (ds, m, "sd_diff"): sd, (ds, m, "rank_biserial"): 1.0, (ds, m, "p_raw"): .000977, (ds, m, "p_holm"): .008789})
    table("tab:app_wilcoxon_random", rows=rnd_rows, tex_rows=rtex, sources=["statistical_tests.compute_final_statistical_report (bootstrap/ + bootstrap_postfix/)"],
          producer="src/statistical_tests.py", bold_rule="none", old_cells=old, keycols=("dataset", "kge"), nd=6)
    for label, key, hist, src in (("tab:app_wilcoxon_deterministic", "DET", HIST["C3"], "milestone_c_h_asym"),
                                  ("tab:app_wilcoxon_sde", "SDE", HIST["C4"], "sde_b6_wilcoxon")):
        grows, gtex = [], []
        for x in [x for x in rep if key in x["family_id"]]:
            per = load(f"progress/{src}_{x['dataset']}.json")["per_seed"]
            d = [p["delta_f1"] for p in per]
            ci = bootstrap_ci(d, rng_seed=12345)
            sg = (sum(v > 0 for v in d), sum(v < 0 for v in d), sum(v == 0 for v in d))
            grows.append({"dataset": x["dataset"], **{k: x[k] for k in ("mean_diff", "sd_diff", "rank_biserial", "p_raw", "p_holm")},
                          "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"], "signs": "/".join(map(str, sg)), "bootstrap_seed": 12345})
            lo = fs(ci["ci_lo"], 6) if ci["ci_lo"] < 0 else f(ci["ci_lo"], 6)
            hi = fs(ci["ci_hi"], 6) if ci["ci_hi"] < 0 else f(ci["ci_hi"], 6)
            gtex.append(f"{NAME[x['dataset']]} & ${fs(x['mean_diff'], 6)}$ & ${f(x['sd_diff'], 6)}$ & $[{lo},{hi}]$ & ${fs(x['rank_biserial'], 3)}$ & "
                        f"${f(x['p_raw'], 6)}$ & ${f(x['p_holm'], 6)}$ & {'/'.join(map(str, sg))} \\\\")
        old = {}
        for ds, (mu, sd, ci, rb, p, ph, signs) in hist.items():
            old.update({(ds, "mean_diff"): mu, (ds, "sd_diff"): sd, (ds, "ci_lo"): ci[0], (ds, "ci_hi"): ci[1], (ds, "rank_biserial"): rb,
                        (ds, "p_raw"): p, (ds, "p_holm"): ph, (ds, "signs"): f"{signs[0]}/{signs[1]}/0"})
        table(label, rows=grows, tex_rows=gtex, sources=[f"progress/{src}_<ds>.json", "statistical_tests.compute_final_statistical_report"],
              producer="src/statistical_tests.py (exact two-sided Wilcoxon, r_rb = (W+ - W-)/(W+ + W-), Holm-3; bootstrap CI seed 12345)",
              bold_rule="none", old_cells=old, keycols=("dataset",), nd=6)


# ── static tables (validated, not regenerated) ─────────────────────────────
def static_checks():
    from data import DataLoader
    from run_all import get_constraint_config
    out = {}
    exp3 = {"FB15k-237": (310116, 14541, 237), "codex-m": (206205, 17050, 51), "WN18RR": (93003, 40943, 11)}
    t3 = {}
    for ds, (n, ne, nr) in exp3.items():
        tr = DataLoader().load_dataset(ds)
        got = (len(tr), len({t.subject for t in tr} | {t.object for t in tr}), len({t.relation for t in tr}))
        t3[ds] = {"got": got, "printed": (n, ne, nr), "match": got == (n, ne, nr)}
    out["tab:Table3Dataset"] = t3
    exp_ode = {"FB15k-237": dict(embedding_dim=50, lambda_data=0.5, lambda_logic=0.1, lambda_asym=1.0, margin_asym=1.0, lambda_reg=0.01),
               "codex-m": dict(embedding_dim=50, lambda_data=10.0, lambda_logic=2.0, lambda_asym=0.5, margin_asym=1.0, lambda_reg=0.1),
               "WN18RR": dict(embedding_dim=50, lambda_data=0.5, lambda_logic=0.05, lambda_asym=1.0, margin_asym=0.5, lambda_reg=0.01)}
    t_ode = {}
    for ds, e in exp_ode.items():
        bp = json.loads((ROOT / ".cache/hyperparams" / f"{ds}_ode.json").read_text())["best_params"]
        t_ode[ds] = {k: {"cache": bp[k], "printed": v, "match": bp[k] == v} for k, v in e.items()}
        t_ode[ds]["T"] = {"cache": bp["t_span"][1], "printed": {"WN18RR": 10.0}.get(ds, 5.0), "match": bp["t_span"][1] == {"WN18RR": 10.0}.get(ds, 5.0)}
    out["tab:app_ode_training"] = t_ode
    exp_c = {"WN18RR": (4, 4, 7, 0), "FB15k-237": (2, 2, 9, 0), "codex-m": (2, 2, 9, 0)}
    t_c = {}
    for ds, v in exp_c.items():
        c = get_constraint_config(ds)
        got = tuple(len(c[k]) for k in ("transitivity_relations", "symmetry_relations", "asymmetry_relations", "cardinality_constraints"))
        t_c[ds] = {"got": got, "printed": v, "match": got == v}
    out["tab:app_constraints"] = t_c
    return out


def non_table_numbers(auroc):
    e = load("progress/e1_e5_evidence_extracted.json")
    jac = [x["jaccard"] for x in e["E1"]["rows"]]
    pr = [abs(x["pearson_rho"]) for x in e["E5"]["rows"]]
    a2 = {ds: loadgz(f"canonical_persist/detections/{ds}/native/native_rk4_50__A2.json.gz") for ds in DS}
    return {
        "E1_jaccard": {"n": len(jac), "min": min(jac), "max": max(jac), "median": float(np.median(jac)), "mean": float(np.mean(jac)),
                       "printed": {"n": 9, "min": .008, "max": .051, "median": .021, "mean": .023}},
        "E4_BC": {x["dataset"]: x["bc_post_fix"] for x in e["E4"]["rows"]} | {"printed": {"FB15k-237": .323, "codex-m": .347, "WN18RR": .464}},
        "E5_abs_pearson": {"mean": float(np.mean(pr)), "median": float(np.median(pr)), "max": max(pr), "printed": {"mean": .129, "median": .123, "max": .258}},
        "E2": {"comparisons": len(e["E2"]["rows"]), "max_abs_cliffs_delta": max(abs(x["cliffs_delta"]) for x in e["E2"]["rows"]),
               "printed": {"comparisons": 135, "max_abs_cliffs_delta": .94}},
        "native_knee": {ds: a2[ds]["threshold"] for ds in DS} | {"printed": {"FB15k-237": 3.776, "codex-m": 2.054, "WN18RR": 7.388}},
        "AUROC_range": auroc | {"printed": {"auroc_populated": 108, "auroc_min": .051, "auroc_max": .918}},
    }


# ── outputs ────────────────────────────────────────────────────────────────
def tex_envs():
    s = TEX.read_text()
    envs = {}
    for lab in TABLES:
        i = s.find("\\label{" + lab + "}")
        if i < 0:
            continue
        a = s.rfind("\\begin{table", 0, i)
        b = s.find("\\end{table", i)
        b = s.find("}", b) + 1
        envs[lab] = s[a:b]
    order = sorted(envs, key=lambda l: s.find("\\label{" + l + "}"))
    return envs, order, s


def replace_body(env, rows):
    m = env.find("\\midrule")
    e = env.find("\\bottomrule")
    return env[:m] + "\\midrule\n" + "\n".join(rows) + "\n" + env[e:]


def main():
    builders = [table5, lambda: logical_table("tab:Table6Logical", "asymmetry", HIST["T6"], False),
                lambda: logical_table("tab:Table7Logical", "transitivity", HIST["T7"], True),
                lambda: logical_table("tab:Table8Logical", "cardinality", HIST["T8"], False),
                table4, table12, weight_sensitivity, progressive, appendix_a, operator_swap, table9_and_stats]
    pending = []
    for b in builders:
        try:
            b()
        except Pending as p:
            pending.append(str(p))
        except ValueError as e:  # statistical_tests refuses incomplete seed sets
            if not ALLOW_PENDING:
                raise
            pending.append(f"{getattr(b, '__name__', 'builder')}: {e}")
    try:
        auroc = table10_and_prevalence()
    except Pending as p:
        pending.append(str(p)); auroc = {}
    if pending and not ALLOW_PENDING:
        raise SystemExit(f"missing canonical inputs: {pending}")
    static = static_checks()
    try:
        extra_numbers = non_table_numbers(auroc)
    except Pending as p:
        pending.append(str(p)); extra_numbers = {}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "csv").mkdir(exist_ok=True)
    if PUBLIC:
        for lab, t in TABLES.items():
            rows = t["rows"]
            with open(OUT / "csv" / f"{lab.replace(':', '_')}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list({k: None for r_ in rows for k in r_}))
                w.writeheader(); w.writerows(rows)
        payload = {"tables": {lab: {"rows": t["rows"], "sources": t["sources"], "producer": t["producer"], "bold_rule": t["bold_rule"],
                                    **({"extra": t["extra"]} if "extra" in t else {})} for lab, t in TABLES.items()},
                   "configuration_checks": {k: {d: (v["got"] if "got" in v else {kk: vv["cache"] for kk, vv in v.items()}) for d, v in x.items()}
                                            for k, x in static.items()},
                   "non_table_numbers": {k: ({kk: vv for kk, vv in v.items() if kk != "printed"} if isinstance(v, dict) else v)
                                         for k, v in extra_numbers.items()},
                   "pending_inputs": pending}
        (OUT / "TABLE_VALUES.json").write_text(json.dumps(payload, indent=1, default=str))
        print(f"tables: {len(TABLES)}; pending: {pending}")
        return
    envs, order, tex_src = tex_envs()
    manifest = (ROOT / "CANONICAL_CODE_COMMIT").read_text().strip()
    values, tex_blocks, md, mapping, delta = {}, [], [], [], []
    import re as _re0
    all_labels = [m_.group(1) for m_ in _re0.finditer(r"\\label\{(tab:[^}]+)\}", tex_src)]
    for lab in order:
        n = all_labels.index(lab) + 1  # manuscript order among all 30 tables
        t = TABLES[lab]
        keycols = t.get("keycols", ("dataset", "method"))
        rows = t["rows"]
        with open(OUT / "csv" / f"{lab.replace(':', '_')}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list({k: None for r_ in rows for k in r_}))
            w.writeheader(); w.writerows(rows)
        # numeric deltas against the printed values
        changes, maxabs, compared = [], 0.0, 0
        for key, oldv in t.get("old_cells", {}).items():
            *kv, col = key if isinstance(key, tuple) else (key,)
            match = [r_ for r_ in rows if tuple(norm(r_.get(k)) for k in keycols) == tuple(norm(x) for x in kv)]
            if not match or col not in match[0] or match[0][col] is None:
                continue
            newv = match[0][col]
            compared += 1
            if isinstance(oldv, str) or isinstance(newv, str):
                if str(oldv) != str(newv):
                    changes.append((key, oldv, newv))
                continue
            if not same_at_printed(oldv, newv):
                changes.append((key, oldv, newv)); maxabs = max(maxabs, abs(float(newv) - float(oldv)))
        # winners: bolded cells, old (printed) vs new
        old_env = envs[lab]
        new_env = replace_body(old_env, t["tex_rows"])
        if t.get("restructured"):
            new_env = new_env.replace("\\begin{tabularx}{\\textwidth}{CCC}",
                                      "% AUTHOR_TEXT_UPDATE_REQUIRED: body restructured, see NUMERIC_DELTA_REPORT.md\n\\begin{tabularx}{\\textwidth}{llCCCC}", 1)
            hdr_a = new_env.find("\\toprule") + len("\\toprule")
            hdr_b = new_env.find("\\midrule")
            new_env = new_env[:hdr_a] + "\n\\textbf{Dataset} & \\textbf{Model} & $\\boldsymbol{d}$ & $\\boldsymbol{\\eta}$ & $\\boldsymbol{\\gamma}$ & \\textbf{Negatives} \\\\\n" + new_env[hdr_b:]
        old_b = old_env[old_env.find("\\midrule"):old_env.find("\\bottomrule")]
        new_b = new_env[new_env.find("\\midrule"):new_env.find("\\bottomrule")]

        def bolded(body):
            import re
            out = []
            for i_, line in enumerate(body.splitlines()):
                for j_, cell in enumerate(line.split("&")):
                    if "\\textbf{" in cell or "\\mathbf{" in cell:
                        out.append((i_, j_))
            return out
        winner_change = bolded(old_b) != bolded(new_b)
        status = "REGENERATE_CANONICAL"
        tex_blocks.append(f"% ===== {lab} =====\n{new_env}\n")
        values[lab] = {"order": n, "status": status, "rows": rows, "sources": t["sources"], "producer": t["producer"],
                       "bold_rule": t["bold_rule"], "n_cells_compared_with_printed": compared, "n_numeric_changes": len(changes),
                       "max_abs_change": maxabs, "winner_changes": winner_change, **({"extra": t["extra"]} if "extra" in t else {}),
                       **({"restructured": t["restructured"]} if t.get("restructured") else {})}
        md.append(f"## {n}. `{lab}`\n\nSource: {', '.join(t['sources'])}  \nBolding: {t['bold_rule']}\n\n" + md_table(rows))
        cap = old_env[old_env.find("\\caption{") + 9:][:90].replace("\n", " ")
        req = bool(changes) or winner_change or bool(t.get("restructured"))
        mapping.append({"table_order": n, "latex_label": lab, "caption_short": cap, "section": section_of(tex_src, lab), "status": status,
                        "producer": t["producer"], "canonical_source": "; ".join(t["sources"]), "csv_path": f"csv/{lab.replace(':', '_')}.csv",
                        "latex_snippet_location": f"TABLES_CANONICAL.tex (% ===== {lab} =====)", "old_values_available": bool(t.get("old_cells")),
                        "numeric_changes": len(changes), "winner_changes": winner_change, "requires_author_text_update": req})
        delta.append((lab, changes, maxabs, winner_change, compared, t))
    # Non-regenerated tables (status from the Phase-0 inventory, next to the output directory)
    inv = OUT.parent / "TABLE_SCHEMA_INVENTORY.md"
    if inv.exists():
        import re as _re
        notes = {"tab:Table14Pairwi": "FROZEN_HUMAN: expert annotations; not recomputed (HUMAN_STUDY_UNTOUCHED)",
                 "tab:Table15Peran": "FROZEN_HUMAN: expert annotations; not recomputed (HUMAN_STUDY_UNTOUCHED)",
                 "tab:app_synthetic": "VALIDATE_STATIC: synthetic illustration independent of the KGs; producer not located in the repository",
                 "tab:app_symbols": "EDITORIAL_ONLY", "tab:Table1Detecti": "NOT_COMPUTATIONAL", "tab:Table2Thethr": "NOT_COMPUTATIONAL",
                 "tab:app_environment": "NOT_COMPUTATIONAL"}
        for line in inv.read_text().splitlines():
            m = _re.match(r"\| (\d+) \| `(tab:[^`]+)` \| ([^|]+) \|.*\| (\w+) \|$", line)
            if not m or m.group(2) in values:
                continue
            order_, lab_, sec_, st_ = int(m.group(1)), m.group(2), m.group(3).strip(), m.group(4)
            sv = static.get(lab_)
            ok = None if sv is None else all((v.get("match") if "match" in v else all(x.get("match", True) for x in v.values())) for v in sv.values())
            note = notes.get(lab_, "") or ("VALIDATE_STATIC: " + ("all printed values match the canonical inputs" if ok else "MISMATCH, see NUMERIC_DELTA_REPORT.md"))
            mapping.append({"table_order": order_, "latex_label": lab_, "caption_short": "", "section": sec_, "status": st_,
                            "producer": note, "canonical_source": "", "csv_path": "", "latex_snippet_location": "manuscript (unchanged)",
                            "old_values_available": True, "numeric_changes": 0 if ok in (True, None) else "see report",
                            "winner_changes": False, "requires_author_text_update": ok is False or lab_ in FACT_NOTES})
        mapping.sort(key=lambda m_: int(m_["table_order"]))
    payload = {"execution_root": str(ROOT), "code_commit": manifest, "tables": values, "static_validation": static,
               "non_table_numbers": extra_numbers, "pending_inputs": pending}
    (OUT / "TABLE_VALUES_CANONICAL.json").write_text(json.dumps(payload, indent=1, default=str))
    (OUT / "TABLES_CANONICAL.tex").write_text("% Canonical tables (one block per \\label). Captions, labels and column formats as in the manuscript;\n"
                                             "% only tabular bodies are regenerated. No prose.\n\n" + "\n".join(tex_blocks))
    (OUT / "TABLES_FOR_AUTHOR.md").write_text("# Canonical tables\n\n" + "\n".join(md))
    with open(OUT / "TABLE_MAPPING.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mapping[0])); w.writeheader(); w.writerows(mapping)
    write_delta(delta, static, extra_numbers)
    print(f"tables: {len(values)}; pending: {pending}")


FACT_NOTES = {
    "tab:app_ode_training": ["The 'Device: MPS' row: in the canonical execution the A1/A2 milestone grid, native A2, H_asym (DET), SDE, "
                             "2x2 and S0-S4 families run on CPU (script defaults); the ODE inside IntegratedKGDebugger "
                             "(panel capture, weight sensitivity) runs on MPS. Parameter values are unchanged."],
    "tab:app_constraints": ["CoDEx-Medium antisymmetry: get_constraint_config lists 10 relations; the table prints 9 (known since Phase B)."],
    "tab:app_complex_training": ["All historical ComplEx results were trained with MarginRankingLoss(margin=1): PyKEEN 1.11.1 pipeline() "
                                 "resolves loss=None to MarginRankingLoss, not to ComplEx.loss_default. Canonical ComplEx uses SoftplusLoss "
                                 "(passed explicitly since 2240c14), as Appendix A states."],
    "tab:Table4Native": ["Native A2 is one set at 50 RK4 steps, shared by Table 4, Table 12 and E1-E5 (the manuscript mixed 100 steps in "
                         "Table 4 with 50 steps in Table 12)."],
    "tab:Table12Tripar": ["Consensus uses the persisted native A2 set of Table 4 (50 RK4 steps) and the four canonical native KGE sets; "
                          "the knee thresholds quoted in the caption change accordingly."],
    "tab:Table5Random": ["Random-substitution evaluation samples now have the nominal prevalence (about 5/10/15%); previously the "
                         "evaluation sample was nested in the perturbed positions (46.5-100% positives)."],
    "tab:app_kge_training": ["Device row: TransE/DistMult/MuRE train on MPS, ComplEx on CPU. MPS training is not bitwise reproducible "
                             "run to run (AUTHOR_REPRODUCIBILITY_LIMITATION_MPS); see PREFLIGHT_REPORT.md."],
}


def norm(v):
    return repr(float(v)) if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) else str(v)


def same_at_printed(old, new):
    """Equal at the precision of the printed value (its own number of decimals)."""
    if isinstance(old, int):
        return int(round(float(new))) == old
    exp = Decimal(str(old)).normalize().as_tuple().exponent
    nd = max(0, -exp)
    return r(new, nd) == r(old, nd)


def section_of(src, lab):
    i = src.find("\\label{" + lab + "}")
    j = max(src.rfind("\\section{", 0, i), src.rfind("\\subsection{", 0, i), src.rfind("\\subsubsection{", 0, i))
    return src[j:src.find("}", j) + 1].split("{", 1)[1].rstrip("}") if j >= 0 else ""


def md_table(rows):
    if not rows:
        return "\n"
    cols = list({k: None for r_ in rows for k in r_})
    def c(v):
        return f"{v:.6g}" if isinstance(v, float) else ("" if v is None else str(v))
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(c(r_.get(k)) for k in cols) + " |" for r_ in rows) + "\n"


def write_delta(delta, static, extra):
    L = ["# Numeric delta report: canonical vs printed values", "",
         "Printed values: constants of the historical validator (`validate_paper_numbers.py`), i.e. the numbers printed in "
         "`KG-Debug-ODE_v4_final_rev1.tex`. A cell counts as changed when the canonical value differs from the printed value at the "
         "printed precision. Notes marked AUTHOR_TEXT_UPDATE_REQUIRED state facts only; no manuscript text is proposed.", ""]
    for lab, changes, maxabs, wc, compared, t in delta:
        L.append(f"## `{lab}`")
        L.append(f"- cells compared with printed values: {compared}; changed: {len(changes)}; max |change|: {maxabs:.4g}; "
                 f"bolded (winner) cells changed: {'yes' if wc else 'no'}")
        if t.get("restructured"):
            L.append(f"- AUTHOR_TEXT_UPDATE_REQUIRED: {t['restructured']}")
        for note in FACT_NOTES.get(lab, []):
            L.append(f"- AUTHOR_TEXT_UPDATE_REQUIRED: {note}")
        if changes or wc:
            L.append("- AUTHOR_TEXT_UPDATE_REQUIRED: every sentence quoting values or rankings from this table must be checked against the canonical values.")
        for key, o, n in changes[:400]:
            L.append(f"  - {key}: printed {o} → canonical {n:.6g}" if isinstance(n, float) else f"  - {key}: printed {o} → canonical {n}")
        L.append("")
    L.append("## Static tables (validated, not regenerated)")
    for lab, d in static.items():
        bad = {k: v for k, v in d.items() if not (v.get("match") if "match" in v else all(x.get("match", True) for x in v.values()))}
        L.append(f"- `{lab}`: {'all printed values match the canonical inputs' if not bad else 'MISMATCH ' + json.dumps(bad, default=str)}")
        for note in FACT_NOTES.get(lab, []):
            L.append(f"  - AUTHOR_TEXT_UPDATE_REQUIRED: {note}")
    L.append("")
    L.append("## Numbers outside table environments")
    for k, v in extra.items():
        L.append(f"- {k}: {json.dumps(v, default=lambda x: round(x, 6) if isinstance(x, float) else str(x))}")
    (OUT / "NUMERIC_DELTA_REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
