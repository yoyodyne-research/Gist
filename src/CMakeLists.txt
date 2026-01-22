include_directories (${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/../libs/kiss_fft130)

option (USE_ARM_NEON "Enable ARM NEON optimizations" OFF)
option (USE_PFFFT "Use PFFFT (NEON/SIMD) backend" OFF)
option (USE_KISS_FFT "Use KissFFT backend" ON)
option (USE_FFTW "Use FFTW backend" OFF)
option (USE_ACCELERATE_FFT "Use Apple Accelerate FFT backend" OFF)
option (USE_BITSTREAM_PITCH "Enable bitstream autocorrelation pitch prototype" OFF)

add_library (
    Gist STATIC
    AccelerateFFT.cpp
    AccelerateFFT.h
    CoreFrequencyDomainFeatures.cpp
    CoreFrequencyDomainFeatures.h
    CoreTimeDomainFeatures.cpp
    CoreTimeDomainFeatures.h
    Gist.cpp
    Gist.h
    MFCC.cpp
    MFCC.h
    OnsetDetectionFunction.cpp
    OnsetDetectionFunction.h
    WindowFunctions.cpp
    WindowFunctions.h
    Yin.cpp
    Yin.h
    BitstreamPitch.h
)

source_group (Source src)

math(EXPR _num_backends \
    "${USE_KISS_FFT}+${USE_PFFFT}+${USE_FFTW}+${USE_ACCELERATE_FFT}")
if (NOT _num_backends EQUAL 1)
    message(FATAL_ERROR "Select exactly one FFT backend: USE_KISS_FFT, USE_PFFFT, USE_FFTW, or USE_ACCELERATE_FFT")
endif()

if (USE_KISS_FFT)
    target_compile_definitions (Gist PUBLIC -DUSE_KISS_FFT)
endif()

if (USE_FFTW)
    target_compile_definitions (Gist PUBLIC -DUSE_FFTW)
endif()

if (USE_ACCELERATE_FFT)
    target_compile_definitions (Gist PUBLIC -DUSE_ACCELERATE_FFT)
endif()

if (USE_ARM_NEON)
    target_compile_definitions (Gist PUBLIC -DUSE_ARM_NEON)
    # Apply conservative NEON flags when building on ARM with GCC/Clang
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        if (CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|armv7l")
            # For armv7 hard-float targets (e.g., Bela/BBB)
            target_compile_options(Gist PRIVATE -mfpu=neon -mcpu=cortex-a8 -mfloat-abi=hard)
        endif()
    endif()
endif()

if (USE_PFFFT)
    target_compile_definitions(Gist PUBLIC -DUSE_PFFFT)
    target_link_libraries(Gist PUBLIC pffft)
    target_include_directories(Gist PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../libs/pffft)
endif()

if (USE_BITSTREAM_PITCH)
    target_compile_definitions(Gist PUBLIC -DUSE_BITSTREAM_PITCH)
endif()
