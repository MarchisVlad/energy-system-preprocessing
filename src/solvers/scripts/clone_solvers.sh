#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SOLVERS_DIR="$SCRIPT_DIR/.."

mkdir -p "$SOLVERS_DIR"

SOLVERS_DIR="$(cd "$SOLVERS_DIR" && pwd)"

REPOS=(
    "https://github.com/NCKempke/PIPS-IPMpp.git"
)

echo "Cloning into: $SOLVERS_DIR"

cd "$SOLVERS_DIR"

for repo in "${REPOS[@]}"; do
    echo "Cloning $repo ..."
    git clone "$repo"
done

echo "All clones complete."
