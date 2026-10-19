#include "selective_scan.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "parallel.h"

namespace mamba_cpu {

namespace {

// Matches torch.nn.functional.softplus with its default threshold of 20.
inline float softplus(float x) { return x <= 20.f ? std::log1p(std::exp(x)) : x; }

inline float silu(float x) { return x / (1.f + std::exp(-x)); }

}  // namespace

void selective_scan_fwd_f32(const SelectiveScanParams& p, int num_threads) {
  const int64_t L = p.seqlen;
  const int64_t N = p.dstate;
  const int64_t dims_per_group = p.dim / p.ngroups;

  // Don't spawn more threads than there is work for (decode-sized calls).
  const int64_t total_work = p.batch * p.dim * L * N;
  const int max_useful = static_cast<int>(std::max<int64_t>(1, total_work / (1 << 15)));
  num_threads = std::min(num_threads, max_useful);

  // Each (batch, channel) pair is an independent recurrence over time.
  parallel_for(0, p.batch * p.dim, num_threads, [&](int64_t lo, int64_t hi) {
    std::vector<float> h(N);
    for (int64_t idx = lo; idx < hi; ++idx) {
      const int64_t b = idx / p.dim;
      const int64_t d = idx % p.dim;
      const int64_t g = d / dims_per_group;

      const float* u = p.u + idx * L;
      const float* delta = p.delta + idx * L;
      const float* z = p.z ? p.z + idx * L : nullptr;
      float* out = p.out + idx * L;
      const float* A = p.A + d * N;
      const float* B = p.B + (b * p.ngroups + g) * L * N;
      const float* C = p.C + (b * p.ngroups + g) * L * N;
      const float D = p.D ? p.D[d] : 0.f;
      const float bias = p.delta_bias ? p.delta_bias[d] : 0.f;

      std::fill(h.begin(), h.end(), 0.f);
      for (int64_t t = 0; t < L; ++t) {
        float dt = delta[t] + bias;
        if (p.delta_softplus) dt = softplus(dt);
        const float dt_u = dt * u[t];
        const float* Bt = B + t * N;
        const float* Ct = C + t * N;
        float y = 0.f;
        for (int64_t n = 0; n < N; ++n) {
          h[n] = std::exp(dt * A[n]) * h[n] + dt_u * Bt[n];
          y += h[n] * Ct[n];
        }
        y += D * u[t];
        if (z) y *= silu(z[t]);
        out[t] = y;
      }
      if (p.last_state) std::copy(h.begin(), h.end(), p.last_state + idx * N);
    }
  });
}

}  // namespace mamba_cpu
