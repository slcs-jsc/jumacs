#!/usr/bin/env bash
#SBATCH --job-name=jumacs-diagnostics
#SBATCH --time=06:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
set -euo pipefail
start="${1:-}"
end="${2:-}"
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
[[ -f pyproject.toml && -d src/jumacs ]] || { echo "Submit from the JuMACS project root" >&2; exit 2; }
period=()
[[ -z $start ]] || period+=(--start-year "$start")
[[ -z $end ]] || period+=(--end-year "$end")
module load Stages/2026 GCCcore/14.3.0 SciPy-Stack/2025b netcdf4-python/1.7.2 BeautifulSoup/4.14.2 PyYAML/6.0.2
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
for model in GEOSCCM EMAC WACCM-X; do
    [[ -d "data/processed/$model" ]] || continue
    [[ -n "$(find "data/processed/$model" -name '*_monthly_zonal.nc' -print -quit)" ]] || continue
    python3 -m jumacs.cli validate --model "$model"
    python3 -m jumacs.cli trends --model "$model"
    python3 -m jumacs.cli quicklook --model "$model" ${period[@]+"${period[@]}"}
done
