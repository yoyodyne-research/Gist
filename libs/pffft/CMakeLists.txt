cmake_minimum_required(VERSION 3.7)
project(pffft C)

add_library(pffft STATIC pffft.c)

target_include_directories(pffft PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|armv7l")
    target_compile_options(pffft PRIVATE -mfpu=neon -mcpu=cortex-a8 -mfloat-abi=hard)
  endif()
endif()
