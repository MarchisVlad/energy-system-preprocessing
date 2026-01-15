#!/usr/bin/env sh

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <pips_root> <utils_path>"
    exit 1
fi

PIPS_ROOT="$1"
UTILS_PATH="$2"

PIPS_ROOT=$(realpath "$PIPS_ROOT")
UTILS_PATH=$(realpath "$UTILS_PATH")

MA27_TAR="ma27-1.0.0.tar.gz"
MA27_SRC="${UTILS_PATH}/${MA27_TAR}"
MA27_DIR="${PIPS_ROOT}/ThirdPartyLibs/MA27"

TOOLCHAIN_FILE="${UTILS_PATH}/Toolchain.cmake"

if [ ! -d "$PIPS_ROOT" ]; then
    echo "Error: pips_root does not exist: $PIPS_ROOT"
    exit 1
fi

if [ ! -f "$MA27_SRC" ]; then
    echo "Error: MA27 archive not found: $MA27_SRC"
    exit 1
fi

if [ ! -f "$TOOLCHAIN_FILE" ]; then
    echo "Error: Toolchain file not found: $TOOLCHAIN_FILE"
    exit 1
fi

if [ ! -d "$MA27_DIR" ]; then
    echo "Error: MA27 directory not found: $MA27_DIR"
    exit 1
fi

echo "Installing MA27..."

echo "Copying ${MA27_TAR} to ${MA27_DIR}"
cp "$MA27_SRC" "$MA27_DIR/"

cd "$MA27_DIR"

if [ ! -x "./installMa27.sh" ]; then
    echo "Error: installMa27.sh not found or not executable"
    exit 1
fi

echo "Running installMa27.sh"
./installMa27.sh

echo "Building PIPS..."

cd "$PIPS_ROOT"

if [ ! -d "build" ]; then
    echo "Creating build directory"
    mkdir build
fi

cd build

echo "Configuring with CMake"
cmake .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE"

echo "Compiling"
make

echo ""
echo "=============================================="
echo "PIPS successfully built"
echo "PIPS root:      $PIPS_ROOT"
echo "Utils path:     $UTILS_PATH"
echo "Toolchain file: $TOOLCHAIN_FILE"
echo "=============================================="
echo ""
