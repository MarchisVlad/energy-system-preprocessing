# Toolchain.cmake - point CMake's FindBoost to the local Boost install
# Replace the path below with your actual boost install prefix
set(BOOST_ROOT "/Users/marchisvlad/energy-system-preprocessing/packages/boost-local" CACHE PATH "Boost root (toolchain)")
set(Boost_NO_SYSTEM_PATHS ON CACHE BOOL "Do not search system paths for Boost")
# Also give CMake a general prefix path to search
set(CMAKE_PREFIX_PATH "${BOOST_ROOT}" CACHE PATH "Prefix path for find_package")
# Optionally prefer static or shared Boost libs:
# set(Boost_USE_STATIC_LIBS ON CACHE BOOL "Use static Boost libs")
# set(Boost_COMPILER "-gcc" CACHE STRING "Optional: specify compiler tag if needed")

# Optional: set RPATH so built executables will look in your Boost lib at runtime
# This embeds rpath into executables produced in this build (Linux)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${BOOST_ROOT}/lib")
set(CMAKE_BUILD_RPATH "${BOOST_ROOT}/lib")
