// Minimal aligned allocation shims for PFFFT on Bela/Linux
// Provides definitions for pffft_aligned_malloc/free expected by pffft.c

#include <stdlib.h>
#include <stdint.h>
#include "pffft.h"

void* pffft_aligned_malloc(size_t nb_bytes) {
    void* ptr = NULL;
    #if defined(__linux__) || defined(__APPLE__) || defined(_POSIX_VERSION)
        // 16-byte alignment is sufficient for NEON/SSE used by PFFFT
        if (posix_memalign(&ptr, 16, nb_bytes) != 0) {
            ptr = NULL;
        }
    #elif defined(_MSC_VER)
        ptr = _aligned_malloc(nb_bytes, 16);
    #else
        // Fallback: overallocate and align manually
        void* raw = malloc(nb_bytes + 16 + sizeof(void*));
        if (!raw) return NULL;
        uintptr_t aligned = ((uintptr_t)raw + sizeof(void*) + 15u) & ~((uintptr_t)15u);
        ((void**)aligned)[-1] = raw;
        ptr = (void*)aligned;
    #endif
    return ptr;
}

void pffft_aligned_free(void* p) {
    if (!p) return;
    #if defined(__linux__) || defined(__APPLE__) || defined(_POSIX_VERSION)
        free(p);
    #elif defined(_MSC_VER)
        _aligned_free(p);
    #else
        void* raw = ((void**)p)[-1];
        free(raw);
    #endif
}

