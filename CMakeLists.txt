cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18)

include("external/madrona/cmake/madrona_init.cmake")

project(MadronaMPEnv LANGUAGES C CXX)

include(setup)
include(dependencies)

add_subdirectory(external)
add_subdirectory(src)
