CARFAC (Cochlear front end) integration stub

This folder is a placeholder for the Google CARFAC C++ sources (Apache-2.0):
  https://github.com/google/carfac

Two ways to vendor:

1) Submodule (recommended)
   git submodule add https://github.com/google/carfac libs/carfac/upstream

   Then either:
   - Add a thin CMakeLists.txt to build the upstream sources, or
   - Copy the minimal .cc/.h you need into libs/carfac/src and include from there.

2) Vendor minimal sources
   Copy the CARFAC C++ headers and sources into libs/carfac/src and keep LICENSE/NOTICE here.

Build integration
- The top-level CMakeLists.txt already calls add_subdirectory(libs/carfac OPTIONAL).
- Provide a libs/carfac/CMakeLists.txt that adds an INTERFACE or STATIC target named carfac.

Bela integration
- The deploy script will copy any files under libs/carfac into the staged project’s carfac/ folder.
- In examples/bela/carfac_frontend.cpp, define HAVE_CARFAC and include the appropriate carfac headers, e.g.:
    #define HAVE_CARFAC 1
    #include "carfac/carfac.h" // adjust to actual path

Licensing
- CARFAC is Apache-2.0. Keep the LICENSE and NOTICE files in this folder.

