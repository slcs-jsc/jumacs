#!/usr/bin/env bash
#SBATCH --job-name=jumacs-zonal-model
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
set -euo pipefail
model="${1:?usage: zonal_model.sh MODEL}"
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
[[ -f pyproject.toml && -d src/jumacs ]] || { echo "Submit from the JuMACS project root" >&2; exit 2; }
module load Stages/2026 GCCcore/14.3.0 SciPy-Stack/2025b netcdf4-python/1.7.2 BeautifulSoup/4.14.2 PyYAML/6.0.2
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
args=()
for variable in "${@:2}"; do
    args+=(--variable "$variable")
done
python3 -m jumacs.cli zonal --model "$model" "${args[@]}"
