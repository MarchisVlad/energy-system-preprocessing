#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SOLVERS_DIR="$SCRIPT_DIR/.."

mkdir -p "$SOLVERS_DIR"
SOLVERS_DIR="$(cd "$SOLVERS_DIR" && pwd)"

# Format:
# "repo_url|clone_folder_name"
REPOS=(
    "https://gitlab.com/pips-ipmpp/detection-annotation|detection_annotation"
)

echo "Cloning into: $SOLVERS_DIR"

cd "$SOLVERS_DIR"

for entry in "${REPOS[@]}"; do
    IFS="|" read -r repo clone_name <<< "$entry"

    echo "Cloning $repo as $clone_name ..."
    git clone "$repo" "$clone_name"
done

echo "All clones complete."
