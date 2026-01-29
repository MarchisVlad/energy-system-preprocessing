#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GENERATORS_DIR="$SCRIPT_DIR/.."

mkdir -p "$GENERATORS_DIR"

GENERATORS_DIR="$(cd "$GENERATORS_DIR" && pwd)"

REPOS=(
    "https://gitlab.com/beam-me/simple-methods.git"
)

echo "Cloning into: $GENERATORS_DIR"

cd "$GENERATORS_DIR"

for repo in "${REPOS[@]}"; do
    echo "Cloning $repo ..."
    git clone "$repo"
done

echo "All clones complete."
