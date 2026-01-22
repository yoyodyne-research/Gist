# Repository Guidelines

## Project Structure & Module Organization
- `src/` — C++ library sources for Gist (core features, FFT wrappers). Builds a static library `Gist`.
- `tests/` — doctest-based unit tests. Entry point `main.cpp`, test files named `Test_*.cpp`, test signals in `tests/test-signals/`.
- `libs/` — third‑party dependencies (e.g., `kiss_fft130/`).
- `python-module/` — optional Python bindings (`setup.py`, `example.py`); see `python-module/INSTALL.md`.
- `documentation/` — Doxygen config and generated docs.
- `.github/workflows/` — CI that builds with CMake and runs `ctest` on major OSes.

## Build, Test, and Development Commands
- Configure and build (default KISS FFT):
  - `mkdir build && cd build`
  - `cmake .. -D BUILD_TESTS=ON`
  - `cmake --build .`
- Run tests: `cd build && ctest -C Debug -VV`
- Select FFT backend (optional CMake definitions):
  - Use KISS FFT: `-DUSE_KISS_FFT` (default in `src/CMakeLists.txt`)
  - Use FFTW: install FFTW, add `-DUSE_FFTW` and link `-lfftw3`
  - Use Apple Accelerate (macOS): `-DUSE_ACCELERATE_FFT`
- Python module (optional): `cd python-module && python3 setup.py build` then `python3 example.py`.

## Coding Style & Naming Conventions
- C++ standard: library targets C++11; tests use C++17 where needed.
- Indentation: 4 spaces; no tabs. Keep line length reasonable (~120 cols).
- Names: Classes/headers PascalCase (e.g., `Gist.h`), methods/functions lowerCamelCase (e.g., `spectralCentroid()`), test files `Test_*.cpp`.
- Headers live alongside sources; include with quotes (`#include "Gist.h"`). Match existing brace/whitespace style.

## Testing Guidelines
- Framework: doctest (header-only, vendored in `tests/doctest/`).
- Add a `Test_<Area>.cpp` per feature; prefer small, focused cases using fixtures from `tests/test-signals/`.
- Run locally via `ctest`; ensure all platforms pass in CI.

## Commit & Pull Request Guidelines
- Commits: imperative, concise subject (<= 72 chars), descriptive body when needed. Group logical changes; update docs/tests with code.
- PRs: include summary, rationale, and notes on FFT backend assumptions. Link related issues; add screenshots only for Python examples if relevant. Ensure `cmake` builds and `ctest` passes.

## Security & Configuration Tips
- Do not commit generated docs, build outputs, or local tool configs. No secrets are used in this repo.
- Prefer the default KISS FFT for portability; document when enabling FFTW/Accelerate and verify linkage on CI.

