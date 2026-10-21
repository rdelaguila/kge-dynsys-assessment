"""Ten paired random-5% seeds for TransE/DistMult/MuRE (Table 5 means, Appendix C.2).

Same code path and output layout as the random sub-experiment of
run_all.run_bootstrap_validation_stage (bootstrap/seed_<s>/random), without the
logical ODE sub-experiments that no KGE table consumes. Requires the native stage.
"""
import argparse
import sys

sys.path.insert(0, "src")
from data import DataLoader  # noqa: E402
from run_all import UnifiedExperimentRunner  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--methods", nargs="+", default=["TransE", "DistMult", "MuRE"])
ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
ap.add_argument("--sample-size", type=int, default=10000)
args = ap.parse_args()

triples = DataLoader().load_dataset(args.dataset)
runner = UnifiedExperimentRunner(args.dataset, args.methods)
for seed in args.seeds:
    print(f"\n--- {args.dataset} random 5% seed {seed} ---", flush=True)
    runner.run_ablation_stage(triples, "random", [5.0], sample_size=args.sample_size,
                              perturbation_seed=seed, output_subdir=f"bootstrap/seed_{seed}/random")
