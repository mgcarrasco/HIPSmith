// Entry point to the program.

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "AbsProgramGenerator.h"
#include "CGOptions.h"
#include "HIPSmith/HIPOptions.h"
#include "HIPSmith/HIPOutputMgr.h"
#include "HIPSmith/HIPProgramGenerator.h"
#include "platform.h"

// Generator seed.
static unsigned long g_Seed = 0;

int main(int argc, char **argv) {
  g_Seed = platform_gen_seed();
  CGOptions::set_default_settings();
  HIPSmith::HIPOptions::ResolveCGOptions();

  for (int idx = 1; idx < argc; ++idx) {
    if (strcmp(argv[idx], "--vectors") == 0) {
      HIPSmith::HIPOptions::vectors(true);
      continue;
    }

    if (strcmp(argv[idx], "--hip-consts") == 0) {
      HIPSmith::HIPOptions::hip_consts(true);
      continue;
    }

    if (strcmp(argv[idx], "--hip-shared") == 0) {
      HIPSmith::HIPOptions::hip_shared(true);
      continue;
    }

    // Allow disabling the guard that forbids taking the address of __shared__
    // memory inside variable initializers. The guard is on by default; this
    // opt-out lets us fuzz the (currently AMDGPU-rejected) pattern on purpose.
    if (strcmp(argv[idx], "--no-hip-shared-safe-static-init") == 0) {
      HIPSmith::HIPOptions::hip_shared_safe_static_init(false);
      continue;
    }

    if (strcmp(argv[idx], "--hip-managed") == 0) {
      HIPSmith::HIPOptions::hip_managed(true);
      continue;
    }

    if (strcmp(argv[idx], "--hip-device") == 0) {
      HIPSmith::HIPOptions::hip_device(true);
      continue;
    }

    if (strcmp(argv[idx], "--hip-builtins") == 0) {
      HIPSmith::HIPOptions::hip_builtins(true);
      continue;
    }

    if (strcmp(argv[idx], "--hip-sync") == 0) {
      HIPSmith::HIPOptions::hip_sync(true);
      continue;
    }

    if (strcmp(argv[idx], "--hip-warp") == 0) {
      HIPSmith::HIPOptions::hip_warp(true);
      continue;
    }

    if (strcmp(argv[idx], "--hip-warp-match") == 0) {
      HIPSmith::HIPOptions::hip_warp_match(true);
      continue;
    }

    if (strcmp(argv[idx], "--hip-warp-shuffle") == 0) {
      HIPSmith::HIPOptions::hip_warp_shuffle(true);
      continue;
    }

    if (strcmp(argv[idx], "--hip-warp-reduce") == 0) {
      HIPSmith::HIPOptions::hip_warp_reduce(true);
      continue;
    }

    if (strcmp(argv[idx], "--atomics") == 0) { 
      HIPSmith::HIPOptions::hip_atomic(true);
      continue;
    }

    if (strcmp(argv[idx], "--small") == 0) {
      HIPSmith::HIPOptions::small(true);
      continue;
    }

    if (strcmp(argv[idx], "--seed") == 0) {
      if (idx + 1 >= argc) {
        std::cerr << "Error: --seed requires a value." << std::endl;
        return -1;
      }

      char *endptr;
      g_Seed = strtoll(argv[idx + 1], &endptr, 10);

      if (*endptr != '\0') {
        std::cerr << "Error: Invalid seed value \"" << argv[idx + 1] << "\""
                  << std::endl;
        return -1;
      }
      idx++;
      continue;
    }

    std::cout << "Invalid option \"" << argv[idx] << "\"" << std::endl;
    return -1;
  }
  // AbsProgramGenerator does other initialisation stuff, besides itself. So
  // we call it, disregarding the returned object. Still need to delete it.
  AbsProgramGenerator *generator =
      AbsProgramGenerator::CreateInstance(argc, argv, g_Seed);
  if (!generator) {
    cout << "error: can't create AbsProgramGenerator. csmith init failed!"
         << std::endl;
    return -1;
  }

  // Create program generator
  HIPSmith::HIPProgramGenerator hip_generator(g_Seed);
  hip_generator.goGenerator();

  // Calls Finalization::doFinalization(), which deletes everything, so must be
  // called after program generation.
  delete generator;

  return 0;
}
