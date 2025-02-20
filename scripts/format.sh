#!/bin/bash
set -Euo pipefail

base_dir=$(dirname $(dirname $0))

# Run the pre-commit checks
SKIP=bandit pre-commit run --all-files

ruff check --config "${base_dir}/pyproject.toml" --fix openfl/

ruff format --config "${base_dir}/pyproject.toml" openfl/