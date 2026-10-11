#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-}"
case "$mode" in
  local)
    source="${JUMACS_HPC_MIRROR:-judac.fz-juelich.de:/p/data1/slmet/model_data/jumacs/site}"
    destination="$root/site"
    ;;
  web) source="$root/site"; destination="${JUMACS_WEB_SITE_MIRROR:-}" ;;
  *) echo "Usage: scripts/mirror.sh {local|web}" >&2; exit 2 ;;
esac

if [[ "$mode" != local && -z "$destination" ]]; then
  echo "Mirror destination is unset or empty" >&2
  exit 2
fi
if [[ "$destination" == / || "$destination" == . || "$destination" == .. || "$destination" == *: || "$destination" == *:/ ]]; then
  echo "Mirror destination is unsafe: $destination" >&2
  exit 2
fi
destination="${destination%/}"
if [[ "$source" != /* && "$source" == *:* ]]; then
  remote_source=true
else
  remote_source=false
fi
if [[ "$remote_source" == false && ! -d "$source" ]]; then
  echo "Mirror source is missing: $source" >&2
  exit 2
fi
if [[ "$mode" == web && ( ! -f "$source/index.html" || ! -d "$source/products/application" ) ]]; then
  echo "Web mirror requires a generated application site with index.html and products/application/" >&2
  exit 2
fi

if [[ "$mode" == local ]]; then
  if [[ "$remote_source" == false && "$(realpath -m "$source")" == "$destination" ]]; then
    echo "Mirror source and destination are the same directory: $source" >&2
    exit 2
  fi
  if [[ ! -d "$root/.git" || -L "$destination" ]]; then
    echo "Run mirror-local from a separate local Git checkout with a regular site/ directory" >&2
    exit 2
  fi
  rsync -av --delete "$source/" "$destination/"
else
  rsync -av --delete "$source/" "$destination/"
fi
