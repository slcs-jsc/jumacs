#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-}"
case "$mode" in
  local)
    hpc_root="$(realpath -m "${JUMACS_HPC_ROOT:-$HOME/jumount/data/slmet/model_data/jumacs}")"
    source="$hpc_root/site"
    destination="$root/site"
    ;;
  web) source="$root/site"; destination="${JUMACS_WEB_SITE_MIRROR:-}" ;;
  data) source="$root/products/release"; destination="${JUMACS_WEB_DATA_MIRROR:-}" ;;
  *) echo "Usage: scripts/mirror.sh {local|web|data}" >&2; exit 2 ;;
esac

if [[ "$mode" != local && -z "$destination" ]]; then
  echo "Mirror destination is unset or empty" >&2
  exit 2
fi
if [[ "$destination" == / || "$destination" == . || "$destination" == .. || "$destination" == *: || "$destination" == *:/ ]]; then
  echo "Mirror destination is unsafe: $destination" >&2
  exit 2
fi
if [[ ! -d "$source" ]]; then
  echo "Mirror source is missing: $source" >&2
  exit 2
fi

if [[ "$mode" == local ]]; then
  if [[ ! -d "$root/.git" || -L "$destination" || "$(realpath -m "$root")" == "$hpc_root" ]]; then
    echo "Run mirror-local from a separate local Git checkout with a regular site/ directory" >&2
    exit 2
  fi
  climatologies=(
    GEOSCCM/jumacs_geosccm_refd1_climatology_1985-2014.nc
    EMAC/jumacs_emac_refd1_climatology_1985-2014.nc
    WACCM-X/jumacs_waccmx_climatology_1985-2014.nc
    combined/jumacs_geosccm_waccmx_1985-2014_5deg_1km.nc
    combined/jumacs_emac_waccmx_1985-2014_5deg_1km.nc
  )
  for relative in "${climatologies[@]}"; do
    if [[ ! -f "$hpc_root/products/climatology/$relative" ]]; then
      echo "Climatology source is missing: $relative" >&2
      exit 2
    fi
  done
  rsync -av --delete "$source/" "$destination/"
  for relative in "${climatologies[@]}"; do
    target="$root/products/climatology/$relative"
    mkdir -p "$(dirname "$target")"
    rsync -av "$hpc_root/products/climatology/$relative" "$target"
  done
else
  rsync -av --delete "$source/" "$destination/"
fi
