cmake_minimum_required (VERSION 3.7.2)
project (Gist VERSION 1.0.7)

set (CMAKE_CXX_STANDARD 11)

option (BUILD_TESTS "Build tests" OFF)

add_subdirectory (src)
add_subdirectory (libs/pffft OPTIONAL)
add_subdirectory (libs/carfac OPTIONAL)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory (tests)
endif (BUILD_TESTS)

set (CMAKE_SUPPRESS_REGENERATION true)
