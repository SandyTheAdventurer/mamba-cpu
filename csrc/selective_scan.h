#pragma once

#include <cstdint>
#include <string>

namespace mamba_cpu {

// Instruction set used by the kernels: "avx2" or "default". Picked at load time
// from the CPU; set_cpu_isa("default" | "avx2" | "auto") overrides it and
// returns false if the requested ISA isn't available.
const char* cpu_isa();
bool set_cpu_isa(const std::string& name);

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

// Backward pass. Reads the forward inputs plus:
//   dout             : (batch, dim, seqlen)
//   dlast_state      : (batch, dim, dstate), may be null
// Writes:
//   du, ddelta, dz   : (batch, dim, seqlen); dz only if z is non-null
//   dB, dC           : (batch, ngroups, seqlen, dstate), must be zero-initialized
//   dA_part          : (batch, dim, dstate)  -- per-batch partials, caller sums over batch
//   dD_part          : (batch, dim)          -- "
//   ddelta_bias_part : (batch, dim)          -- "
struct SelectiveScanBwdParams {
  SelectiveScanParams fwd;  // out and last_state are unused
  const float* dout;
  const float* dlast_state;
  float* du;
  float* ddelta;
  float* dA_part;
  float* dB;
  float* dC;
  float* dD_part;
  float* dz;
  float* ddelta_bias_part;
};

void selective_scan_bwd_f32(const SelectiveScanBwdParams& p, int num_threads);

}  // namespace mamba_cpu
