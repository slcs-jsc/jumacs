#!/usr/bin/env bash
#SBATCH --job-name=jumacs-comparison-plots
#SBATCH --time=01:00:00
#SBATCH --mem=8G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
[[ -f pyproject.toml && -d src/jumacs ]] || { echo "Submit from the JuMACS project root" >&2; exit 2; }
module load Stages/2026 GCCcore/14.3.0 SciPy-Stack/2025b netcdf4-python/1.7.2 BeautifulSoup/4.14.2 PyYAML/6.0.2
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
python3 scripts/plot_comparisons.py
python3 scripts/build_index.py
