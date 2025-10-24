#pragma once

#include <nvshmem.h>

#include "core/cuda_utils.h"

#define NVSHMEMCHECK(stmt)                                                                         \
  do {                                                                                             \
    int result = (stmt);                                                                           \
    if (NVSHMEMX_SUCCESS != result) {                                                              \
      fprintf(stderr, "[%s:%d] nvshmem failed with error %d \n", __FILE__, __LINE__, result);      \
      exit(-1);                                                                                    \
    }                                                                                              \
  } while (0)
