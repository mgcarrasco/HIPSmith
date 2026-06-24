#include "HIPSmith/HIPOptions.h"

#include <iostream>

#include "CGOptions.h"

namespace HIPSmith {

// Eww macro, used here just to make the flags easier to write.
#define DEFINE_HIPFLAG(name, type, init)      \
  type HIPOptions::name##_ = init;            \
  type HIPOptions::name() { return name##_; } \
  void HIPOptions::name(type x) { name##_ = x; }
DEFINE_HIPFLAG(output, const char*, "HIPProg.hip")
DEFINE_HIPFLAG(safe_math, bool, true)
DEFINE_HIPFLAG(small, bool, false)
DEFINE_HIPFLAG(is_emitting_device_code, bool, false)
DEFINE_HIPFLAG(vectors, bool, false)
DEFINE_HIPFLAG(hip_consts, bool, false)
DEFINE_HIPFLAG(hip_shared, bool, false)
DEFINE_HIPFLAG(hip_shared_safe_static_init, bool, true)
DEFINE_HIPFLAG(hip_managed, bool, false)
DEFINE_HIPFLAG(hip_device, bool, false)
DEFINE_HIPFLAG(hip_builtins, bool, false)
DEFINE_HIPFLAG(hip_sync, bool, false)
DEFINE_HIPFLAG(hip_warp, bool, false)
DEFINE_HIPFLAG(hip_warp_match, bool, false)
DEFINE_HIPFLAG(hip_warp_shuffle, bool, false)
DEFINE_HIPFLAG(hip_warp_reduce, bool, false)
DEFINE_HIPFLAG(hip_atomic, bool, false)
#undef DEFINE_HIPFLAG

void HIPOptions::set_default_settings() {
  output_ = "HIPProg.hip";
  safe_math_ = true;
  small_ = false;
  is_emitting_device_code_ = false;
  vectors_ = false;
  hip_consts_ = false;
  hip_shared_ = false;
  hip_shared_safe_static_init_ = true;
  hip_managed_ = false;
  hip_device_ = false;
  hip_builtins_ = false;
  hip_sync_ = false;
  hip_warp_ = false;
  hip_warp_match_ = false;
  hip_warp_shuffle_ = false;
  hip_warp_reduce_ = false;
  hip_atomic_ = false;
}

void HIPOptions::ResolveCGOptions() {
  // General settings for normal HIP programs.
  // No static in OpenCL.
  CGOptions::force_globals_static(false);

  // Maybe enable in future. Has a different syntax.
  CGOptions::packed_struct(false);
  // No printf in OpenCL.
  CGOptions::hash_value_printf(false);
  // The way we currently handle globals means we need to disable consts.
  CGOptions::consts(false);
  // Volatiles were playing up so turned off
  CGOptions::volatiles(false);
  // cpp options
  CGOptions::lang_cpp(true);
  CGOptions::cpp11(true);
  // due to array initialisation problems (a global seed) we make this uniform
  CGOptions::force_non_uniform_array_init(false);

  // Setting for small programs.
  if (small_) {
    // Limit number of functions to no more than 3.
    CGOptions::max_funcs(3);
    CGOptions::max_blk_depth(3);
    CGOptions::max_expr_depth(5);
    CGOptions::max_block_size(20);
    CGOptions::max_array_dimensions(3);
    CGOptions::max_array_length_per_dimension(5);
    CGOptions::max_array_length(7);
  }
}

}  // namespace HIPSmith
