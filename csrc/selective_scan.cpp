#include "selective_scan.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <vector>

#include "parallel.h"

namespace mamba_cpu {

namespace {

// Matches torch.nn.functional.softplus with its default threshold of 20.
inline float softplus(float x) { return x <= 20.f ? std::log1p(std::exp(x)) : x; }

inline float silu(float x) { return x / (1.f + std::exp(-x)); }

inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

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

void selective_scan_bwd_f32(const SelectiveScanBwdParams& p, int num_threads) {
  const SelectiveScanParams& f = p.fwd;
  const int64_t L = f.seqlen;
  const int64_t N = f.dstate;
  const int64_t dims_per_group = f.dim / f.ngroups;

  const int64_t total_work = f.batch * f.dim * L * N;
  const int max_useful = static_cast<int>(std::max<int64_t>(1, total_work / (1 << 14)));
  num_threads = std::min(num_threads, max_useful);

  // dB/dC are shared by every channel of a (batch, group) slice. Threads own
  // contiguous channel ranges, so each accumulates its current slice locally
  // and flushes it once under that slice's lock.
  std::vector<std::mutex> slice_locks(f.batch * f.ngroups);

  parallel_for(0, f.batch * f.dim, num_threads, [&](int64_t lo, int64_t hi) {
    std::vector<float> hs(L * N);  // h_t for every t, recomputed per channel
    std::vector<float> dB_acc(L * N), dC_acc(L * N);
    std::vector<float> gh(N), dA_acc(N);
    int64_t cur_slice = -1;

    auto flush = [&] {
      if (cur_slice < 0) return;
      std::lock_guard<std::mutex> lock(slice_locks[cur_slice]);
      float* dB = p.dB + cur_slice * L * N;
      float* dC = p.dC + cur_slice * L * N;
      for (int64_t i = 0; i < L * N; ++i) {
        dB[i] += dB_acc[i];
        dC[i] += dC_acc[i];
      }
    };

    for (int64_t idx = lo; idx < hi; ++idx) {
      const int64_t b = idx / f.dim;
      const int64_t d = idx % f.dim;
      const int64_t slice = b * f.ngroups + d / dims_per_group;
      if (slice != cur_slice) {
        flush();
        cur_slice = slice;
        std::fill(dB_acc.begin(), dB_acc.end(), 0.f);
        std::fill(dC_acc.begin(), dC_acc.end(), 0.f);
      }

      const float* u = f.u + idx * L;
      const float* delta = f.delta + idx * L;
      const float* z = f.z ? f.z + idx * L : nullptr;
      const float* dout = p.dout + idx * L;
      float* du = p.du + idx * L;
      float* ddelta = p.ddelta + idx * L;
      float* dz = p.dz ? p.dz + idx * L : nullptr;
      const float* A = f.A + d * N;
      const float* B = f.B + slice * L * N;
      const float* C = f.C + slice * L * N;
      const float D = f.D ? f.D[d] : 0.f;
      const float bias = f.delta_bias ? f.delta_bias[d] : 0.f;

      auto get_dt = [&](int64_t t) {
        const float dt = delta[t] + bias;
        return f.delta_softplus ? softplus(dt) : dt;
      };

      // Forward recompute, keeping every state.
      for (int64_t t = 0; t < L; ++t) {
        const float dt = get_dt(t);
        const float dt_u = dt * u[t];
        const float* Bt = B + t * N;
        const float* h_prev = t > 0 ? &hs[(t - 1) * N] : nullptr;
        float* h = &hs[t * N];
        for (int64_t n = 0; n < N; ++n) {
          h[n] = std::exp(dt * A[n]) * (h_prev ? h_prev[n] : 0.f) + dt_u * Bt[n];
        }
      }

      // Reverse scan. gh holds dLoss/dh_t, carried back through h_t = a * h_{t-1} + dt*u*B.
      if (p.dlast_state) {
        std::copy(p.dlast_state + idx * N, p.dlast_state + (idx + 1) * N, gh.begin());
      } else {
        std::fill(gh.begin(), gh.end(), 0.f);
      }
      std::fill(dA_acc.begin(), dA_acc.end(), 0.f);
      float dD_acc = 0.f;
      float dbias_acc = 0.f;

      for (int64_t t = L - 1; t >= 0; --t) {
        const float dt = get_dt(t);
        const float dt_u = dt * u[t];
        const float* Bt = B + t * N;
        const float* Ct = C + t * N;
        const float* h = &hs[t * N];
        const float* h_prev = t > 0 ? &hs[(t - 1) * N] : nullptr;
        float* dBt = &dB_acc[t * N];
        float* dCt = &dC_acc[t * N];

        // out = y * silu(z), y = <h, C> + D*u
        float dy = dout[t];
        if (z) {
          float y = D * u[t];
          for (int64_t n = 0; n < N; ++n) y += h[n] * Ct[n];
          const float s = sigmoid(z[t]);
          dz[t] = dout[t] * y * s * (1.f + z[t] * (1.f - s));
          dy = dout[t] * z[t] * s;
        }
        dD_acc += dy * u[t];

        float ddt = 0.f;
        float gh_dot_B = 0.f;
        for (int64_t n = 0; n < N; ++n) {
          const float g = gh[n] + dy * Ct[n];
          dCt[n] += dy * h[n];
          const float a = std::exp(dt * A[n]);
          const float g_a = g * (h_prev ? h_prev[n] : 0.f) * a;  // dLoss/d(dt*A[n])
          dA_acc[n] += g_a * dt;
          ddt += g_a * A[n] + g * u[t] * Bt[n];
          gh_dot_B += g * Bt[n];
          dBt[n] += g * dt_u;
          gh[n] = g * a;
        }
        du[t] = dy * D + dt * gh_dot_B;
        if (f.delta_softplus) ddt *= sigmoid(delta[t] + bias);
        ddelta[t] = ddt;
        dbias_acc += ddt;
      }

      std::copy(dA_acc.begin(), dA_acc.end(), p.dA_part + idx * N);
      p.dD_part[idx] = dD_acc;
      p.ddelta_bias_part[idx] = dbias_acc;
    }
    flush();
  });
}

}  // namespace mamba_cpu
