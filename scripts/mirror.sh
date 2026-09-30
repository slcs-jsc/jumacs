#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-}" in
  local) source="$root/site"; destination="${JUMACS_LOCAL_SITE_MIRROR:-}" ;;
  web) source="$root/site"; destination="${JUMACS_WEB_SITE_MIRROR:-}" ;;
  data) source="$root/products/release"; destination="${JUMACS_WEB_DATA_MIRROR:-}" ;;
  *) echo "Usage: scripts/mirror.sh {local|web|data}" >&2; exit 2 ;;
esac

if [[ -z "$destination" ]]; then
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

rsync -av --delete "$source/" "$destination/"
