#pragma once

#include <cstdint>

namespace mamba_cpu {

// All tensors are contiguous float32. Shapes:
//   u, delta, z, out : (batch, dim, seqlen)
//   A                : (dim, dstate)
//   B, C             : (batch, ngroups, seqlen, dstate)   -- note: seqlen before dstate
//   D, delta_bias    : (dim)
//   last_state       : (batch, dim, dstate)
// D, z, delta_bias and last_state may be null.
struct SelectiveScanParams {
  const float* u;
  const float* delta;
  const float* A;
  const float* B;
  const float* C;
  const float* D;
  const float* z;
  const float* delta_bias;
  float* out;
  float* last_state;
  int64_t batch;
  int64_t dim;
  int64_t seqlen;
  int64_t dstate;
  int64_t ngroups;
  bool delta_softplus;
};

void selective_scan_fwd_f32(const SelectiveScanParams& p, int num_threads);

}  // namespace mamba_cpu
