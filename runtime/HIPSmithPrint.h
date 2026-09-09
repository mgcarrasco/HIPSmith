#pragma once

// Definitions of the PRINT_* macro family emitted by StatementPrint.
//
// Every PRINT_* macro has the signature (v, lineno, how, id) and expands one of
// three ways, selected at compile time so that a single generated program can
// be built in all three modes without regenerating it:
//
//   (default)                 printf the value, one line per print
//   -DHIPSMITH_PRINT_NOOP     expand to ((void)0)
//   -DHIPSMITH_PRINT_ESCAPE   force the variable's storage to survive, no output
//
// Included by HIPSmith.h, so generated code needs no extra include.

#include <cstdio>

#include <hip/hip_runtime.h>

#if defined(HIPSMITH_PRINT_NOOP) && defined(HIPSMITH_PRINT_ESCAPE)
#error "HIPSMITH_PRINT_NOOP and HIPSMITH_PRINT_ESCAPE are mutually exclusive"
#endif

#ifdef HIPSMITH_PRINT_NOOP
#define PRINT_INT8(v, lineno, how, id) ((void)0)
#define PRINT_UINT8(v, lineno, how, id) ((void)0)
#define PRINT_INT16(v, lineno, how, id) ((void)0)
#define PRINT_UINT16(v, lineno, how, id) ((void)0)
#define PRINT_INT(v, lineno, how, id) ((void)0)
#define PRINT_UINT(v, lineno, how, id) ((void)0)
#define PRINT_INT64(v, lineno, how, id) ((void)0)
#define PRINT_UINT64(v, lineno, how, id) ((void)0)
#elif defined(HIPSMITH_PRINT_ESCAPE)
#define HIPSMITH_ESCAPE(v) ((void)*(const volatile decltype(v) *)&(v))
#define PRINT_INT8(v, lineno, how, id) HIPSMITH_ESCAPE(v)
#define PRINT_UINT8(v, lineno, how, id) HIPSMITH_ESCAPE(v)
#define PRINT_INT16(v, lineno, how, id) HIPSMITH_ESCAPE(v)
#define PRINT_UINT16(v, lineno, how, id) HIPSMITH_ESCAPE(v)
#define PRINT_INT(v, lineno, how, id) HIPSMITH_ESCAPE(v)
#define PRINT_UINT(v, lineno, how, id) HIPSMITH_ESCAPE(v)
#define PRINT_INT64(v, lineno, how, id) HIPSMITH_ESCAPE(v)
#define PRINT_UINT64(v, lineno, how, id) HIPSMITH_ESCAPE(v)
#else
#ifdef __HIP_DEVICE_COMPILE__
#define HIPSMITH_PRINT(v, lineno, how, id, fmt, val)                           \
  printf("[line %d] tid(%u,%u,%u) %s = " fmt " how='%s' id=%d sizeof=%u\n",     \
         (int)(lineno), (unsigned)threadIdx.x, (unsigned)threadIdx.y,          \
         (unsigned)threadIdx.z, #v, val, how, (int)(id),                       \
         (unsigned)sizeof(v))
#else
#define HIPSMITH_PRINT(v, lineno, how, id, fmt, val)                           \
  printf("[line %d] %s = " fmt " how='%s' id=%d sizeof=%u\n", (int)(lineno),    \
         #v, val, how, (int)(id), (unsigned)sizeof(v))
#endif
#define PRINT_INT8(v, lineno, how, id)                                         \
  HIPSMITH_PRINT(v, lineno, how, id, "%d", (int)(v))
#define PRINT_UINT8(v, lineno, how, id)                                        \
  HIPSMITH_PRINT(v, lineno, how, id, "%u", (unsigned)(v))
#define PRINT_INT16(v, lineno, how, id)                                        \
  HIPSMITH_PRINT(v, lineno, how, id, "%d", (int)(v))
#define PRINT_UINT16(v, lineno, how, id)                                       \
  HIPSMITH_PRINT(v, lineno, how, id, "%u", (unsigned)(v))
#define PRINT_INT(v, lineno, how, id)                                          \
  HIPSMITH_PRINT(v, lineno, how, id, "%d", (int)(v))
#define PRINT_UINT(v, lineno, how, id)                                         \
  HIPSMITH_PRINT(v, lineno, how, id, "%u", (unsigned)(v))
#define PRINT_INT64(v, lineno, how, id)                                        \
  HIPSMITH_PRINT(v, lineno, how, id, "%lld", (long long)(v))
#define PRINT_UINT64(v, lineno, how, id)                                       \
  HIPSMITH_PRINT(v, lineno, how, id, "%llu", (unsigned long long)(v))
#endif
