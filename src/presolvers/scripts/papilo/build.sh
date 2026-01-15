#!/usr/bin/env sh

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <papilo_root> <utils_path>"
    exit 1
fi

PAPILO_ROOT="$1"
UTILS_PATH="$2"

PAPILO_ROOT=$(realpath "$PAPILO_ROOT")
UTILS_PATH=$(realpath "$UTILS_PATH")
TOOLCHAIN_FILE="${UTILS_PATH}/Toolchain.cmake"

if [ ! -d "$PAPILO_ROOT" ]; then
    echo "Error: PAPILO_root does not exist: $PAPILO_ROOT"
    exit 1
fi

if [ ! -f "$TOOLCHAIN_FILE" ]; then
    echo "Error: Toolchain file not found: $TOOLCHAIN_FILE"
    exit 1
fi

echo "Building PaPILO..."

cd "$PAPILO_ROOT"

if [ ! -d "build" ]; then
    echo "Creating build directory"
    mkdir build
fi

cd build

echo "Configuring with CMake"
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE"

echo "Compiling"
make

echo ""
echo "=============================================="
echo "PaPILO successfully built"
echo "PaPILO root:      $PAPILO_ROOT"
echo "Utils path:     $UTILS_PATH"
echo "Toolchain file: $TOOLCHAIN_FILE"
echo "=============================================="
echo ""
