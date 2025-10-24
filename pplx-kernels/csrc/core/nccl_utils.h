#pragma once

#define NCCLCHECK(cmd)                                                                             \
  do {                                                                                             \
    ncclResult_t r = cmd;                                                                          \
    if (r != ncclSuccess) {                                                                        \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));        \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)
