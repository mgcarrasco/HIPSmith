# HIPSmith

## About

HIPSmith is a random generator of AMD HIP programs, built on top of [CSmith](https://github.com/csmith-project/csmith). Its primary purpose is to find bugs in HIP compilers (like AMD's ROCm/HIP toolchain) using differential testing as the test oracle.

Like its parent project CSmith, HIPSmith outputs programs free of undefined behaviors, but tailored for the heterogeneous computing environment of GPUs and CPUs.

## Install HIPSmith

You can build HIPSmith from source. The following commands assume a standard Linux environment (e.g., Ubuntu).

### Prerequisites
* Git
* C++ Compiler (GCC or Clang)
* CMake (Version 2.8.12 or higher)
* M4 (Required for generating safe math macros)
* Python 3 (For running the test script)

### Build Instructions

1.  **Create a build directory (Out-of-source build recommended):**
    ```bash
    mkdir build
    cd build
    ```

2.  **Configure and Build:**
    ```bash
    # Configure the project
    cmake ..

    # Compile
    make
    ```
    *Note: `cmake ..` will automatically copy `csmith.h`, `HIPSmith.h`, `HIPSmithPrint.h`, and generate `safe_math_macros.h` into your build directory.*

## Use HIPSmith

The primary way to use HIPSmith for differential testing is via the provided Python script, `hip_test.py`. This script automates the process of generating random HIP programs, compiling them, and executing them to find inconsistencies.

### Running Automated Tests

To start the testing loop:

```bash
python3 test_hipsmith.py
```

### HIPSmith Command-Line Options

HIPSmith accepts the following HIP-specific flags:

| Flag | Description |
|---|---|
| `--seed <N>` | Fix the random seed for reproducible output |
| `--small` | Restrict output size (max 3 functions, shallower blocks/expressions/arrays) |
| `--vectors` | Enable HIP vector types |
| `--atomics` | Enable atomic operations |
| `--hip-consts` | Generate global read-only `__constant__` variables |
| `--hip-shared` | Generate `__shared__` local memory variables |
| `--no-hip-shared-safe-static-init` | Disable the restrictions that avoid unsupported `__shared__` features under `--hip-shared` |
| `--hip-managed` | Generate `__managed__` variables |
| `--no-hip-managed-safe-static-init` | Disable the restrictions that avoid unsupported `__managed__` features under `--hip-managed` |
| `--hip-device` | Generate `__device__` variables |
| `--hip-builtins` | Use HIP built-in functions (e.g. `threadIdx`, `blockIdx`) |
| `--hip-print` | Randomly emit `PRINT_<TYPE>(lvalue, __LINE__, how, id);` statements |
| `--hip-print-same-line` | Like `--hip-print`, but emit each `PRINT_*` first on the same line as the following statement. Omit a print that has no such neighbor. Implies `--hip-print`. |
| `--hip-sync` | Emit barrier synchronization (`__syncthreads()`, `__threadfence()`, etc.) |
| `--hip-warp` | Enable warp-level operations |
| `--hip-warp-match` | Enable warp match operations (`__match_any_sync`, etc.) |
| `--hip-warp-shuffle` | Enable warp shuffle operations (`__shfl_*`) |
| `--hip-warp-reduce` | Enable warp reduction operations |
| `--hip-argc-threads` | Derive `num_threads` and `block_size` from `argc`, forcing a single thread |

#### Scalar print statements

`--hip-print` emits `PRINT_<TYPE>(lvalue, __LINE__, how, id);` (~10% of
statements). `how` is a ROCgdb `print` expression;
`id` is a stable generation-order integer. The macro prints `sizeof(lvalue)`
for gdb comparison; that size is not a source operand, so a reducer cannot
rewrite it.

The expansion is chosen at compile time, so one generated program can be built
in any of these modes without regenerating:

| Define | `PRINT_*` expands to |
|---|---|
| *(none)* | `printf` of the value, `sizeof`, `how` and `id` |
| `-DHIPSMITH_PRINT_NOOP` | `((void)0)` |
| `-DHIPSMITH_PRINT_ESCAPE` | a discarded volatile read of the variable (device only, no output) |

#### Grid and block size behaviour

By default, the generated driver launches the kernel with `num_threads = 4` and `block_size = 4`.

Several flags cause the driver to derive these values from `argc` instead, so that the binary can be run with no extra arguments (`argc = 1`) to enforce single-thread / single-block execution and avoid data races or undefined thread-dependent behaviour:

| Flag | `num_threads = argc` | `block_size = argc` |
|---|---|---|
| `--hip-shared` | | yes |
| `--hip-managed` | yes | yes |
| `--hip-device` | yes | yes |
| `--hip-builtins` | yes | yes |
| `--hip-sync` | yes | yes |
| `--hip-warp` | yes | yes |
| `--hip-warp-match` | yes | yes |
| `--hip-warp-shuffle` | yes | yes |
| `--hip-warp-reduce` | yes | yes |
| `--atomics` | yes | yes |
| `--hip-argc-threads` | yes | yes |