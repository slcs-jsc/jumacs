#!/usr/bin/env bash
#SBATCH --job-name=jumacs-zonal
#SBATCH --time=06:00:00
#SBATCH --mem=16G
set -euo pipefail
model="${1:?usage: zonal_array.sh MODEL VARIABLE}"
variable_or_list="${2:?variable or variable-list file required}"
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
[[ -f pyproject.toml && -d src/jumacs ]] || { echo "Submit from the JuMACS project root" >&2; exit 2; }
module load Stages/2026 GCCcore/14.3.0 SciPy-Stack/2025b netcdf4-python/1.7.2 BeautifulSoup/4.14.2 PyYAML/6.0.2
if [[ -f "$variable_or_list" ]]; then
    index="${SLURM_ARRAY_TASK_ID:?array index required for a variable list}"
    variable="$(sed -n "$((index + 1))p" "$variable_or_list")"
    [[ -n "$variable" ]] || { echo "No variable at array index $index" >&2; exit 2; }
else
    variable="$variable_or_list"
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m jumacs.cli zonal --model "$model" --variable "$variable"
