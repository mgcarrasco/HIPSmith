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
    *Note: `cmake ..` will automatically copy `csmith.h`, `HIPSmith.h`, and generate `safe_math_macros.h` into your build directory.*

## Use HIPSmith

The primary way to use HIPSmith for differential testing is via the provided Python script, `hip_test.py`. This script automates the process of generating random HIP programs, compiling them, and executing them to find inconsistencies.

### Running Automated Tests

To start the testing loop:

```bash
python3 hip_test.py
```

### HIPSmith Command-Line Options

HIPSmith accepts the following HIP-specific flags:

| Flag | Description |
|---|---|
| `--seed <N>` | Fix the random seed for reproducible output |
| `--small` | Restrict output size (max 3 functions, shallower blocks/expressions/arrays) |
| `--vectors` | Enable HIP vector types |
| `--hip-consts` | Generate global read-only `__constant__` variables |
| `--hip-shared` | Generate `__shared__` local memory variables |
| `--hip-managed` | Generate `__managed__` variables |
| `--hip-device` | Generate `__device__` variables |
| `--hip-builtins` | Use HIP built-in functions (e.g. `threadIdx`, `blockIdx`) |
| `--hip-sync` | Emit barrier synchronization (`__syncthreads()`, `__threadfence()`, etc.) |
| `--hip-warp` | Enable warp-level operations |
| `--hip-warp-match` | Enable warp match operations (`__match_any_sync`, etc.) |
| `--hip-warp-shuffle` | Enable warp shuffle operations (`__shfl_*`) |
| `--hip-warp-reduce` | Enable warp reduction operations |

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