#!/usr/bin/env python3
"""Submit one zonal job per model, then dependent climatology and comparison."""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = {
    "GEOSCCM": ROOT / "products/manifests/geosccm_refd1.json",
    "EMAC": ROOT / "products/manifests/emac_refd1.json",
}


def submit(command):
    print(" ".join(str(x) for x in command), flush=True)
    result = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    return result.stdout.strip().split(";", 1)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", required=True)
    parser.add_argument("--start-year", type=int, default=1985)
    parser.add_argument("--end-year", type=int, default=2014)
    parser.add_argument("--execute", action="store_true", help="Submit jobs; otherwise show readiness")
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("Invalid period")
    plans = {}
    for model, path in MANIFESTS.items():
        manifest = json.loads(path.read_text())
        if manifest["model"] != model or manifest["experiment"] != "refD1":
            parser.error("Incorrect manifest: {}".format(path))
        if args.start_year < manifest["start_year"] or args.end_year > manifest["end_year"]:
            parser.error("Climatology period lies outside the downloaded manifest: {}".format(model))
        pending = [row for row in manifest["files"] if row["download_status"] != "complete" or not (ROOT / row["local_path"]).is_file()]
        variables = sorted({row["variable"] for row in manifest["files"]})
        print("{}: {} variables, {} pending raw files".format(model, len(variables), len(pending)))
        plans[model] = pending
    if not args.execute:
        print("Dry run only. Submit after both pending counts reach zero, using --execute.")
        return
    if any(plans.values()):
        parser.error("Raw download is incomplete; no jobs submitted")
    log_dir = ROOT / "cache/slurm"
    log_dir.mkdir(parents=True, exist_ok=True)
    climate_jobs = []
    for model in MANIFESTS:
        zonal = submit(["sbatch", "--parsable", "-A", args.account, "--export=ALL",
                        "--output={}/%x-%j.out".format(log_dir),
                        str(ROOT / "scripts/slurm/zonal_model.sh"), model])
        climate = submit(["sbatch", "--parsable", "-A", args.account,
                          "--export=ALL", "--dependency=afterok:{}".format(zonal),
                          "--output={}/%x-%j.out".format(log_dir),
                          str(ROOT / "scripts/slurm/climatology.sh"), model,
                          str(args.start_year), str(args.end_year)])
        climate_jobs.append(climate)
        print("{} zonal {}, climatology {}".format(model, zonal, climate))
    comparison = submit(["sbatch", "--parsable", "-A", args.account,
                         "--export=ALL", "--dependency=afterok:{}".format(":".join(climate_jobs)),
                         "--output={}/%x-%j.out".format(log_dir),
                         str(ROOT / "scripts/slurm/compare.sh"),
                         str(args.start_year), str(args.end_year)])
    print("Comparison job {}".format(comparison))
    diagnostics = submit(["sbatch", "--parsable", "-A", args.account,
                          "--export=ALL", "--dependency=afterok:{}".format(comparison),
                          "--output={}/%x-%j.out".format(log_dir),
                          str(ROOT / "scripts/slurm/diagnostics.sh"),
                          str(args.start_year), str(args.end_year)])
    print("Validation and plots job {}".format(diagnostics))


if __name__ == "__main__":
    main()
