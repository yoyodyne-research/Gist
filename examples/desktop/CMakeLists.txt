# Desktop build for Kitsune latent layer testing
# Enables local development and testing without deploying to Bela

cmake_minimum_required(VERSION 3.14)
project(kitsune_desktop VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Desktop-specific flag (disables Bela-specific code paths)
add_compile_definitions(KITSUNE_DESKTOP=1)

# Find Eigen (required for CARFAC)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# CARFAC library (from libs/)
set(CARFAC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../libs/carfac/upstream/cpp")
add_library(carfac STATIC
    ${CARFAC_DIR}/car.cc
    ${CARFAC_DIR}/carfac.cc
    ${CARFAC_DIR}/ear.cc
)
target_include_directories(carfac PUBLIC
    ${CARFAC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../libs/carfac
)
target_link_libraries(carfac PUBLIC Eigen3::Eigen)
target_compile_definitions(carfac PUBLIC HAVE_CARFAC=1)

# Gist library (from src/)
set(GIST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
add_library(gist STATIC
    ${GIST_DIR}/Gist.cpp
    ${GIST_DIR}/CoreTimeDomainFeatures.cpp
    ${GIST_DIR}/CoreFrequencyDomainFeatures.cpp
    ${GIST_DIR}/OnsetDetectionFunction.cpp
    ${GIST_DIR}/MFCC.cpp
    ${GIST_DIR}/WindowFunctions.cpp
    ${GIST_DIR}/Yin.cpp
)
target_include_directories(gist PUBLIC ${GIST_DIR})

# Kitsune test executable
set(BELA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../bela")
add_executable(kitsune_test
    main.cpp
    ${BELA_DIR}/carfac_frontend.cpp
    ${BELA_DIR}/latent_layer.cpp
)

target_include_directories(kitsune_test PRIVATE
    ${BELA_DIR}
    ${CARFAC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../libs/carfac
    ${CMAKE_CURRENT_SOURCE_DIR}/../../libs
)

target_link_libraries(kitsune_test
    carfac
)

target_compile_definitions(kitsune_test PRIVATE
    KITSUNE_DESKTOP=1
    HAVE_CARFAC=1
)

# Optional: Enable sanitizers for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(kitsune_test PRIVATE -fsanitize=address -fsanitize=undefined)
    target_link_options(kitsune_test PRIVATE -fsanitize=address -fsanitize=undefined)
endif()

# Unit test for latent layer only (no audio processing)
add_executable(latent_test
    latent_test.cpp
    ${BELA_DIR}/latent_layer.cpp
)

target_include_directories(latent_test PRIVATE
    ${BELA_DIR}
)

target_compile_definitions(latent_test PRIVATE
    KITSUNE_DESKTOP=1
)

# Install
install(TARGETS kitsune_test latent_test
    RUNTIME DESTINATION bin
)
