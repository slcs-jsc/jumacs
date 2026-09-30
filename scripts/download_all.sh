#!/usr/bin/env bash

# Check or complete all JuMACS archive downloads.
#
# Usage:
#   scripts/download_all.sh
#   scripts/download_all.sh --execute
#
# Default:
#   refresh inventories, regenerate full-archive manifests and report download status only.
#
# --execute:
#   additionally download all pending files.
#
# The script is restartable. Existing complete files are skipped.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
cd "${ROOT}" || exit 1

EXECUTE=false

case "${1:-}" in
    "")
        ;;
    --execute)
        EXECUTE=true
        ;;
    *)
        echo "Usage: $0 [--execute]"
        exit 1
        ;;
esac

# Set up the JUWELS Python environment in this script's shell.
# shellcheck disable=SC1091
source scripts/python_setup.sh || exit 1

mapfile -t MODELS < <(
    python - <<'PY'
from jumacs.config import models_with_capability
for model in models_with_capability("download"):
    print(model)
PY
)

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "ERROR: No download-capable models found."
    exit 1
fi

echo
echo "JuMACS archive download check"
echo "============================="
echo "Models: ${#MODELS[@]}"
echo "Mode:   $([[ "${EXECUTE}" == true ]] && echo execute || echo check-only)"
echo

FAILED=()

for model in "${MODELS[@]}"; do
    echo
    echo "================================================================"
    echo "${model}"
    echo "================================================================"

    echo "Refreshing archive inventory from CEDA (metadata only)..."
    if ! python -m jumacs.cli inspect \
        --model "${model}" \
        > /dev/null
    then
        echo "ERROR: Could not inspect ${model}."
        FAILED+=("${model}:inspect")
        continue
    fi

    echo "Creating/updating full archive manifest..."
    if ! python -m jumacs.cli download \
        --model "${model}" \
        --all-files
    then
        echo "ERROR: Could not create manifest for ${model}."
        FAILED+=("${model}:manifest")
        continue
    fi

    if [[ "${EXECUTE}" == true ]]; then
        echo
        echo "Checking and completing download..."
        if ! python scripts/archive_download.py \
            --model "${model}" \
            --execute
        then
            echo "WARNING: Download incomplete for ${model}."
            echo "It can be resumed by running this script again."
            FAILED+=("${model}:download")
            continue
        fi
    else
        echo
        python scripts/archive_download.py --model "${model}" || {
            FAILED+=("${model}:check")
            continue
        }
    fi
done

echo
echo "================================================================"
echo "Final status"
echo "================================================================"

# Re-check every model without downloading.
INCOMPLETE=()

for model in "${MODELS[@]}"; do
    if ! output="$(python scripts/archive_download.py --model "${model}" 2>&1)"; then
        echo "${model}: ERROR"
        echo "  ${output}"
        INCOMPLETE+=("${model}")
        continue
    fi

    echo "${output}"

    if ! grep -qE ', 0 pending,' <<< "${output}"; then
        INCOMPLETE+=("${model}")
    fi
done

echo

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Operations with errors:"
    printf '  %s\n' "${FAILED[@]}"
    echo
fi

if [[ ${#INCOMPLETE[@]} -gt 0 ]]; then
    echo "Still incomplete:"
    printf '  %s\n' "${INCOMPLETE[@]}"
    echo
    echo "Run again to continue:"
    echo "  $0 --execute"
    exit 1
fi

echo "All download-capable JuMACS sources are complete."
