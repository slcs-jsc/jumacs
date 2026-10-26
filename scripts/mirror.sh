#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-}"
dry_run=false
if [[ $# -gt 2 || ( $# -eq 2 && "$2" != --dry-run ) ]]; then
  echo "Usage: scripts/mirror.sh {local|web} [--dry-run]" >&2
  exit 2
fi
if [[ "${2:-}" == --dry-run ]]; then
  dry_run=true
fi
case "$mode" in
  local)
    source="${JUMACS_HPC_MIRROR:-judac.fz-juelich.de:/p/data1/slmet/model_data/jumacs/site}"
    destination="$root/site"
    ;;
  web) source="$root/site"; destination="${JUMACS_WEB_SITE_MIRROR:-}" ;;
  *) echo "Usage: scripts/mirror.sh {local|web} [--dry-run]" >&2; exit 2 ;;
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
if [[ "$mode" == web && ( ! -f "$source/index.html" || ! -d "$source/products/harmonized" ) ]]; then
  echo "Web mirror requires a generated site with index.html and products/harmonized/" >&2
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
fi

rsync_args=(-av --delete)
if [[ "$dry_run" == true ]]; then
  rsync_args+=(--dry-run)
  echo "Dry run: nothing will be transferred or deleted in $destination" >&2
fi
rsync "${rsync_args[@]}" "$source/" "$destination/"
