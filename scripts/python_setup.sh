#!/usr/bin/env bash
set -euo pipefail

# JuMACS Python environment setup for JUWELS.
#
# Usage:
#   source ./scripts/python_setup.sh            # create/activate the venv
#   source ./scripts/python_setup.sh --update   # also refresh pip, JuMACS and dependencies
#
# The script:
#   - loads the JUWELS Python module
#   - creates .venv if needed (installing JuMACS with the test extra)
#   - activates the environment in the current shell
#   - with --update, upgrades the editable install and its dependencies

if [[ -z "${BASH_VERSION:-}" ]]; then
    echo "Please run this script with bash."
    return 1 2>/dev/null || exit 1
fi

# Must be sourced so that module/venv changes remain active.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Please source this script:"
    echo
    echo "  source $0"
    exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

update=false
if [[ "${1:-}" == "--update" ]]; then
    update=true
elif [[ -n "${1:-}" ]]; then
    echo "Unknown option: $1 (usage: source ${BASH_SOURCE[0]} [--update])"
    return 1
fi

# JUWELS environment modules.
if command -v module >/dev/null 2>&1; then
    module load Python/3.13.5
else
    echo "ERROR: Environment Modules not available."
    return 1
fi

echo "Using $(python3 --version)"

VENV="${ROOT}/.venv"

# Recreate the environment if it does not exist or uses the wrong Python.
recreate=false

if [[ ! -x "${VENV}/bin/python" ]]; then
    recreate=true
else
    venv_version="$("${VENV}/bin/python" -c \
        'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    current_version="$(python3 -c \
        'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

    if [[ "${venv_version}" != "${current_version}" ]]; then
        echo "Existing .venv uses Python ${venv_version}; current module provides ${current_version}."
        recreate=true
    fi
fi

if [[ "${recreate}" == true ]]; then
    echo "Creating JuMACS virtual environment..."
    rm -rf "${VENV}"
    python3 -m venv "${VENV}"
fi

source "${VENV}/bin/activate"

if [[ "${recreate}" == true ]]; then
    echo "Installing JuMACS with test extras..."
    python -m pip install --upgrade pip
    python -m pip install -e ".[test]"
elif [[ "${update}" == true ]]; then
    echo "Updating JuMACS and dependencies..."
    python -m pip install --upgrade -e ".[test]"
fi

echo
echo "JuMACS environment ready:"
echo "  Python: $(python --version)"
echo "  Venv:   ${VIRTUAL_ENV}"
echo
echo "Example:"
echo "  python -m jumacs.cli inspect --model all"
