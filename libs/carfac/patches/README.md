# CARFAC local patches

The `libs/carfac/upstream` submodule tracks https://github.com/google/carfac,
which we cannot push to. Local modifications required by this project live here
as patch files and must be re-applied after a fresh clone / submodule update:

```sh
git submodule update --init libs/carfac/upstream
git -C libs/carfac/upstream apply ../patches/0001-add-mutable-ear-accessor.patch
```

## Patches

- `0001-add-mutable-ear-accessor.patch` — adds `CARFAC::mutable_ear()`, used by
  `examples/bela/carfac_frontend.cpp` for direct per-sample processing without
  `RunSegment` overhead.
