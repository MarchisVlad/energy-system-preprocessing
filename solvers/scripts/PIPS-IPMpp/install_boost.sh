#!/usr/bin/env sh

set -e

# Check arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 <install_directory>"
    exit 1
fi

TARGET_DIR="$1"
TARGET_DIR=$(realpath "$TARGET_DIR")
BOOST_LOCAL_DIR="${TARGET_DIR}/boost-local"

BOOST_VERSION="1.78.0"
BOOST_DIR="boost_$(echo "$BOOST_VERSION" | tr '.' '_')"
BOOST_TAR="${BOOST_DIR}.tar.gz"
BOOST_URL="https://archives.boost.io/release/${BOOST_VERSION}/source/${BOOST_TAR}"


# Ensure target directory exists
echo "Creating target directory (if not exists): $TARGET_DIR"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# Download Boost
echo "Downloading Boost ${BOOST_VERSION}"
wget "$BOOST_URL" -O "$BOOST_TAR"

# Extract
echo "Extracting ${BOOST_TAR}"
tar xf "$BOOST_TAR"

# Build + install Boost
echo "Bootstrapping Boost..."
cd "$BOOST_DIR"
./bootstrap.sh --prefix="${BOOST_LOCAL_DIR}"

echo "Building and installing Boost..."
./b2

cd ..

# Cleanup
echo "Cleaning up..."
rm -f "$BOOST_TAR"
# rm -rf "$BOOST_DIR"

# echo ""
echo "=============================================="
echo "Boost ${BOOST_VERSION} successfully installed at:"
echo "    $BOOST_LOCAL_DIR"
echo "=============================================="
echo ""
