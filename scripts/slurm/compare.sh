#!/usr/bin/env bash
#SBATCH --job-name=jumacs-compare
#SBATCH --time=04:00:00
#SBATCH --mem=16G
set -euo pipefail
start="${1:-}"
end="${2:-}"
first="${3:-GEOSCCM}"
second="${4:-EMAC}"
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
[[ -f pyproject.toml && -d src/jumacs ]] || { echo "Submit from the JuMACS project root" >&2; exit 2; }
module load Stages/2026 GCCcore/14.3.0 SciPy-Stack/2025b netcdf4-python/1.7.2 BeautifulSoup/4.14.2 PyYAML/6.0.2
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
period=()
[[ -z $start ]] || period+=(--start-year "$start")
[[ -z $end ]] || period+=(--end-year "$end")
python3 -m jumacs.cli compare --models "$first" "$second" ${period[@]+"${period[@]}"}
