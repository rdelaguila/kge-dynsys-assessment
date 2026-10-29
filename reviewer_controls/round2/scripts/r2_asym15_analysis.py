"""R2 panel 1 analysis — evaluator coherence, seed-42 runner check, per-seed metrics, exclusive recovery, fusion
contrasts and families F1-F3 (round-2 protocol; see ../README.md).

Run from the repository root:  python reviewer_controls/round2/scripts/r2_asym15_analysis.py <frozen_inputs> <work>
Inputs (read-only): <work>/asym15/*.npz (r2_asym15_run.py), <work>/inputs and <work>/block_a (stage1/stage2),
<work>/r2_2/R2_2_EXCLUSIVE_RECOVERY.csv, frozen progress/20260623 npz and paper_results/*/ablation_audit json.
Output: <work>/asym15_analysis/.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import spearmanr

from r2_common import exclusive, family, fusion, metrics

FROZEN, WK = Path(sys.argv[1]), Path(sys.argv[2])
OUT = WK / "asym15_analysis"
PANEL_SEEDS = list(range(1, 11))
KGE = ("TransE", "DistMult", "MuRE")
TOL = 1e-12


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader(); w.writerows(rows)


def frozen_cell(ds, vt, n):
    a2 = np.load(FROZEN / "progress/20260623" / f"ode_postfix_A2_{ds}_{vt}_{n}.npz", allow_pickle=True)
    a1 = np.load(FROZEN / "progress/20260623" / f"ode_postfix_A1_{ds}_{vt}_{n}.npz", allow_pickle=True)
    audit = json.loads((FROZEN / "paper_results" / ds / "ablation_audit" / f"{vt}_{n}.json").read_text())["triples"]
    inp = np.load(WK / "inputs" / f"{ds}__grid_{vt}_{n}.npz")
    S = {"A2": np.asarray(a2["eval_energies"], float), "A1": np.asarray(a1["eval_energies"], float),
         "degree": inp["degree"].astype(float)}
    for k in KGE:
        S[k] = -np.array([r["kge_raw"][k] for r in audit], float)
    T = [(r["s"], r["r"], r["o"]) for r in audit]
    return T, inp["gt"].astype(bool), S


def coherence():
    """Evaluator must reproduce block_a metrics (nine cells) and R2.2 regions (WN18RR asym-15) from frozen scores."""
    ref = {(r["cell"], r["config"]): r for r in csv.DictReader(open(WK / "block_a/metrics.csv"))}
    cells = sorted({c for c, _ in ref})
    rows = []
    for cell in cells:
        ds, vt, n = cell.split("/")
        _, gt, S = frozen_cell(ds, vt, n)
        cfg = dict(S)
        cfg["Fusion(A2,TransE)"] = fusion(S["A2"], S["TransE"])
        cfg["Fusion(degree,TransE)"] = fusion(S["degree"], S["TransE"])
        for name, s in cfg.items():
            m = metrics(gt, s)
            r = ref[(cell, name)]
            for k in ("ndcg_at_K", "auroc"):
                d = abs(m[k] - float(r[k]))
                rows.append({"check": "block_a_metrics", "cell": cell, "config": name, "metric": k,
                             "recomputed": m[k], "reference": float(r[k]), "abs_diff": d, "ok": d <= TOL})
    _, gt, S = frozen_cell("WN18RR", "asymmetry", 13950)
    r22 = {r["kge"]: r for r in csv.DictReader(open(WK / "r2_2/R2_2_EXCLUSIVE_RECOVERY.csv"))
           if r["dataset"] == "WN18RR" and r["generator"] == "asymmetry"}
    for k in KGE:
        e = exclusive(gt, S["A2"], S[k], "a2")
        for f in ("a2_only_n", "a2_only_tp", "shared_tp", "kge_only_tp", "a2_only_enrichment", "kge_only_enrichment"):
            d = abs(float(e[f]) - float(r22[k][f]))
            rows.append({"check": "r2_2_regions", "cell": "WN18RR/asymmetry/13950", "config": f"A2 vs {k}", "metric": f,
                         "recomputed": e[f], "reference": r22[k][f], "abs_diff": d, "ok": d <= TOL})
    return rows


def load_seed(seed):
    z = np.load(WK / "asym15" / f"WN18RR_asym15_seed{seed}.npz", allow_pickle=False)
    S = {"A2": z["A2"].astype(float), "A1": z["A1"].astype(float)}
    for k in KGE:
        S[k] = -z[f"{k}_raw"].astype(float)
    for k in ("deg_sum", "deg_h", "deg_t"):
        S[k] = z[k].astype(float)
    T = [tuple(t) for t in z["triples"]]
    return T, z["gt"].astype(bool), S, str(z["sha"]), json.loads(str(z["meta"]))


def runner_check():
    """Seed 42 from the new runner versus the frozen artefacts (criteria fixed in the round-2 protocol)."""
    T, gt, S, sha, meta = load_seed(42)
    Tf, gtf, Sf = frozen_cell("WN18RR", "asymmetry", 13950)
    rows = [{"quantity": "sample SHA-256", "runner": sha, "frozen": "b8928d07086781f93bef0774364cc18d4c01760ef8d777d8faffcab4252e5f24",
             "status": "EXACT" if sha.startswith("b8928d07086781f93bef0774364cc18d4c01760ef8d777d8faffcab4252e5f24") else "MISMATCH"},
            {"quantity": "sample order (vs frozen audit)", "runner": len(T), "frozen": len(Tf), "status": "EXACT" if T == Tf else "MISMATCH"},
            {"quantity": "labels per position", "runner": int(gt.sum()), "frozen": int(gtf.sum()),
             "status": "EXACT" if np.array_equal(gt, gtf) else "MISMATCH"},
            {"quantity": "deg_sum vs stage1 degree", "runner": "", "frozen": "",
             "max_abs_diff": float(np.max(np.abs(S["deg_sum"] - Sf["degree"]))),
             "status": "EXACT" if np.array_equal(S["deg_sum"], Sf["degree"]) else "MISMATCH"}]
    for k in ("A2", "A1", *KGE):
        a, b = S[k], Sf[k]
        ma, mb = metrics(gt, a), metrics(gt, b)
        rows.append({"quantity": f"{k} score", "backend": meta.get(f"{k}_device", "cpu"),
                     "max_abs_diff": float(np.max(np.abs(a - b))), "spearman": float(spearmanr(a, b).statistic),
                     "ndcg_runner": ma["ndcg_at_K"], "ndcg_frozen": mb["ndcg_at_K"], "d_ndcg": ma["ndcg_at_K"] - mb["ndcg_at_K"],
                     "auroc_runner": ma["auroc"], "auroc_frozen": mb["auroc"], "d_auroc": ma["auroc"] - mb["auroc"],
                     "status": "EXACT" if np.array_equal(a, b) else "MPS_NUMERICAL_DEVIATION_DOCUMENTED"
                     if k in KGE else "NUMERICAL_DEVIATION"})
    for k in KGE:
        g_run = metrics(gt, fusion(S["A2"], S[k]))["ndcg_at_K"] - metrics(gt, S[k])["ndcg_at_K"]
        g_frz = metrics(gt, fusion(Sf["A2"], Sf[k]))["ndcg_at_K"] - metrics(gt, Sf[k])["ndcg_at_K"]
        rows.append({"quantity": f"Fusion(A2,{k}) - {k} NDCG", "runner": g_run, "frozen": g_frz, "d_ndcg": g_run - g_frz,
                     "status": "EXACT" if g_run == g_frz else "DEVIATION_DOCUMENTED"})
    return rows


def per_seed(seed):
    T, gt, S, sha, meta = load_seed(seed)
    tag = "reference" if seed == 42 else "panel"
    nonfinite = {k: int((~np.isfinite(v)).sum()) for k, v in S.items()}
    cfg = dict(S)
    for k in KGE:
        cfg[f"Fusion(A2,{k})"] = fusion(S["A2"], S[k])
        cfg[f"Fusion(A1,{k})"] = fusion(S["A1"], S[k])
        cfg[f"Fusion(deg_sum,{k})"] = fusion(S["deg_sum"], S[k])
    cfg["Fusion(deg_h,TransE)"] = fusion(S["deg_h"], S["TransE"])
    cfg["Fusion(deg_t,TransE)"] = fusion(S["deg_t"], S["TransE"])
    M = {}
    met_rows = []
    for name, s in cfg.items():
        M[name] = metrics(gt, s)
        met_rows.append({"seed": seed, "role": tag, "sample_sha256": sha, "config": name, "N": len(gt), "P": int(gt.sum()),
                         "prevalence": gt.mean(), "nonfinite": nonfinite.get(name, 0), **M[name]})
    exc_rows = []
    for a in ("A2", "A1"):
        for k in KGE:
            exc_rows.append({"seed": seed, "role": tag, "dynamic": a, "kge": k, **exclusive(gt, S[a], S[k], "dyn")})
    exc_rows.append({"seed": seed, "role": tag, "dynamic": "deg_sum", "kge": "TransE",
                     **exclusive(gt, S["deg_sum"], S["TransE"], "dyn")})
    nd = {k: v["ndcg_at_K"] for k, v in M.items()}
    fus_rows = []
    for k in KGE:
        fus_rows.append({"seed": seed, "role": tag, "kge": k, "ndcg_kge": nd[k], "ndcg_fusion_A2": nd[f"Fusion(A2,{k})"],
                         "ndcg_fusion_degree": nd[f"Fusion(deg_sum,{k})"],
                         "delta_A2": nd[f"Fusion(A2,{k})"] - nd[k], "delta_degree": nd[f"Fusion(deg_sum,{k})"] - nd[k],
                         "delta_specific": nd[f"Fusion(A2,{k})"] - nd[f"Fusion(deg_sum,{k})"],
                         "auroc_delta_A2": M[f"Fusion(A2,{k})"]["auroc"] - M[k]["auroc"],
                         "auroc_delta_degree": M[f"Fusion(deg_sum,{k})"]["auroc"] - M[k]["auroc"]})
    for v in ("deg_h", "deg_t"):
        fus_rows.append({"seed": seed, "role": tag, "kge": "TransE", "degree_variant": v, "ndcg_kge": nd["TransE"],
                         "ndcg_fusion_degree": nd[f"Fusion({v},TransE)"],
                         "delta_degree": nd[f"Fusion({v},TransE)"] - nd["TransE"],
                         "delta_specific": nd["Fusion(A2,TransE)"] - nd[f"Fusion({v},TransE)"]})
    a_rows = []
    for k in KGE:
        a_rows.append({"seed": seed, "role": tag, "kge": k, "ndcg_A2": nd["A2"], "ndcg_A1": nd["A1"], "A2_minus_A1": nd["A2"] - nd["A1"],
                       "delta_A1": nd[f"Fusion(A1,{k})"] - nd[k], "delta_A2": nd[f"Fusion(A2,{k})"] - nd[k],
                       "delta_A1_A2": nd[f"Fusion(A1,{k})"] - nd[f"Fusion(A2,{k})"],
                       "spearman_A1_A2": float(spearmanr(S["A1"], S["A2"]).statistic)})
    # secondary diagnostic: permuted-A2 fusion with TransE (1000 permutations, default_rng(0)), as in stage2_block_a
    rng = np.random.default_rng(0)
    perm = np.array([metrics(gt, fusion(rng.permutation(S["A2"]), S["TransE"]))["ndcg_at_K"] for _ in range(1000)])
    perm_row = {"seed": seed, "role": tag, "fusion_ndcg": nd["Fusion(A2,TransE)"], "perm_mean": perm.mean(),
                "perm_q025": np.quantile(perm, .025), "perm_q975": np.quantile(perm, .975),
                "p_perm": (1 + (perm >= nd["Fusion(A2,TransE)"]).sum()) / 1001}
    return met_rows, exc_rows, fus_rows, a_rows, perm_row, nd, nonfinite


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    coh = coherence()
    write(OUT / "evaluator_coherence.csv", coh)
    bad = [r for r in coh if not r["ok"]]
    print(f"coherence: {len(coh) - len(bad)}/{len(coh)} ok", flush=True)
    if bad:
        sys.exit("evaluator coherence failed; analysis stopped")
    write(OUT / "seed42_runner_check.csv", runner_check())
    allm, alle, allf, alla, allp, ND, NF = [], [], [], [], [], {}, {}
    for seed in [42] + PANEL_SEEDS:
        m, e, f, a, p, nd, nf = per_seed(seed)
        allm += m; alle += e; allf += f; alla += a; allp.append(p); ND[seed] = nd; NF[seed] = nf
    write(OUT / "metrics_per_seed.csv", allm)
    write(OUT / "exclusive_recovery.csv", alle)
    write(OUT / "fusion_comparisons.csv", allf)
    write(OUT / "a1_a2_comparisons.csv", alla)
    write(OUT / "permutation_control.csv", allp)

    def parts(name):
        return name[len("Fusion("):-1].split(",") if name.startswith("Fusion(") else [name]

    def d(a, b):
        """Paired differences over panel seeds whose component scores are all finite (round-2 protocol)."""
        used = [s for s in PANEL_SEEDS if all(NF[s][c] == 0 for c in parts(a) + parts(b))]
        return np.array([ND[s][a] - ND[s][b] for s in used])

    fams = []
    fams += family("F1", [("Fus(A2,TransE) - TransE", d("Fusion(A2,TransE)", "TransE")),
                          ("Fus(deg_sum,TransE) - TransE", d("Fusion(deg_sum,TransE)", "TransE")),
                          ("Fus(A2,TransE) - Fus(deg_sum,TransE)", d("Fusion(A2,TransE)", "Fusion(deg_sum,TransE)"))])
    fams += family("F2", [("Fus(A1,TransE) - TransE", d("Fusion(A1,TransE)", "TransE")),
                          ("Fus(A1,TransE) - Fus(A2,TransE)", d("Fusion(A1,TransE)", "Fusion(A2,TransE)"))])
    fams += family("F3", [("Fus(A2,DistMult) - DistMult", d("Fusion(A2,DistMult)", "DistMult")),
                          ("Fus(A2,MuRE) - MuRE", d("Fusion(A2,MuRE)", "MuRE")),
                          ("Fus(A1,DistMult) - DistMult", d("Fusion(A1,DistMult)", "DistMult")),
                          ("Fus(A1,MuRE) - MuRE", d("Fusion(A1,MuRE)", "MuRE"))])
    write(OUT / "paired_statistics_F1_F3.csv", fams)
    # descriptive summaries (not confirmatory): exclusive enrichment per seed, degree variants, AUROC
    from r2_common import contrast
    desc = []
    for dyn in ("A2", "A1", "deg_sum"):
        for k in KGE:
            vals = [r["dyn_only_enrichment"] for r in alle if r["role"] == "panel" and r["dynamic"] == dyn and r["kge"] == k]
            if not vals:
                continue
            defined = np.array([v for v in vals if v != "NA"], float)
            desc.append({"summary": f"{dyn}-only enrichment vs {k}", "n_seeds": len(vals), "n_defined": len(defined),
                         **({k2: v2 for k2, v2 in contrast(defined - 1.0).items()
                             if k2 in ("mean", "median", "sd", "min", "max", "ci95_lo", "ci95_hi", "n_pos", "n_neg")}
                            if len(defined) > 1 else {}),
                         "note": "statistics on enrichment - 1; descriptive"})
    for v in ("deg_h", "deg_t"):
        for lab, a, b in ((f"Fus({v},TransE) - TransE", f"Fusion({v},TransE)", "TransE"),
                          (f"Fus(A2,TransE) - Fus({v},TransE)", "Fusion(A2,TransE)", f"Fusion({v},TransE)")):
            c = contrast(d(a, b))
            desc.append({"summary": lab, "n_seeds": c["n_available"], **{k2: c[k2] for k2 in
                         ("mean", "median", "sd", "min", "max", "ci95_lo", "ci95_hi", "n_pos", "n_neg", "p_raw")},
                         "note": "descriptive; not part of F1"})
    write(OUT / "descriptive_summaries.csv", desc)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
