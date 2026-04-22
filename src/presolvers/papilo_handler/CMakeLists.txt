cmake_minimum_required(VERSION 3.15)
project(papilo_handler CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point to the already-built PaPILO. papilo-config.cmake handles finding
# Boost, TBB, GMP, and Quadmath transitively from that build.
set(papilo_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../papilo/build"
    CACHE PATH "Directory containing papilo-config.cmake")

find_package(papilo CONFIG REQUIRED)

add_executable(papilo_handler papilo_handler.cpp)
target_link_libraries(papilo_handler PRIVATE papilo-core)
