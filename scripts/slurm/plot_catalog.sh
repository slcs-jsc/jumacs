#!/usr/bin/env bash
#SBATCH --job-name=jumacs-plots
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
module load Stages/2026 GCCcore/14.3.0 SciPy-Stack/2025b netcdf4-python/1.7.2 BeautifulSoup/4.14.2 PyYAML/6.0.2
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3 scripts/plot_catalog.py
