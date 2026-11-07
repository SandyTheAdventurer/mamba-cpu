#include "selective_scan.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "parallel.h"

// Hot loops are written as always-inline "row" bodies. Each body is wrapped
// once per ISA: a baseline wrapper, and on x86 GCC/Clang an AVX2+FMA wrapper
// selected at runtime. Using per-function target attributes (rather than
// compiling a TU with -mavx2) keeps out-of-line copies of shared inline/template
// code at the baseline ISA, so nothing AVX2 leaks into code run on older CPUs.

#if defined(__GNUC__) || defined(__clang__)
#define MAMBA_INLINE inline __attribute__((always_inline))
#define MAMBA_PRAGMA(x) _Pragma(#x)
#define MAMBA_SIMD MAMBA_PRAGMA(omp simd)
#define MAMBA_SIMD_REDUCE(...) MAMBA_PRAGMA(omp simd reduction(+ : __VA_ARGS__))
#elif defined(_MSC_VER)
#define MAMBA_INLINE __forceinline
#define MAMBA_SIMD
#define MAMBA_SIMD_REDUCE(...)
#else
#define MAMBA_INLINE inline
#define MAMBA_SIMD
#define MAMBA_SIMD_REDUCE(...)
#endif

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define MAMBA_X86_DISPATCH 1
#define MAMBA_TARGET_AVX2 __attribute__((target("avx2,fma")))
#else
#define MAMBA_X86_DISPATCH 0
#endif

namespace mamba_cpu {

namespace {

// Matches torch.nn.functional.softplus with its default threshold of 20.
MAMBA_INLINE float softplus(float x) { return x <= 20.f ? std::log1p(std::exp(x)) : x; }

MAMBA_INLINE float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

MAMBA_INLINE float bit_cast_float(int32_t bits) {
#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
  return __builtin_bit_cast(float, bits);
#else
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
#endif
}

// Branch-free expf (Cephes-style, ~1-2 ulp) built only from arithmetic and bit
// casts, so loops calling it auto-vectorize.
//
// x must be clamped before range reduction: for huge |x| the reduction loses all
// precision and the polynomial overflows, which turns a decay of ~0 into NaN.
// GCC only vectorizes these float ternaries with -fno-trapping-math (see
// CMakeLists.txt); without it the code is still correct, just scalar. Results
// saturate to ~1e-38 / ~2^127 rather than 0 / inf, and NaN propagates.
MAMBA_INLINE float exp_approx(float x) {
  x = x < -87.f ? -87.f : x;
  x = x > 88.f ? 88.f : x;
  // Round x*log2(e) to nearest via the 1.5*2^23 trick (no rounding intrinsic needed).
  const float fx = (x * 1.44269504088896341f + 12582912.f) - 12582912.f;
  const float r = x - fx * 0.693359375f + fx * 2.12194440e-4f;
  float y = 1.9875691500e-4f;
  y = y * r + 1.3981999507e-3f;
  y = y * r + 8.3334519073e-3f;
  y = y * r + 4.1665795894e-2f;
  y = y * r + 1.6666665459e-1f;
  y = y * r + 5.0000001201e-1f;
  y = y * r * r + r + 1.f;
  return y * bit_cast_float((static_cast<int32_t>(fx) + 127) * (1 << 23));
}

struct RowPtrs {
  const float* u;
  const float* delta;
  const float* z;
  const float* A;
  const float* B;  // (seqlen, dstate) slice for this row's (batch, group)
  const float* C;
  float D;
  float bias;
};

MAMBA_INLINE RowPtrs row_ptrs(const SelectiveScanParams& p, int64_t idx) {
  const int64_t L = p.seqlen;
  const int64_t N = p.dstate;
  const int64_t b = idx / p.dim;
  const int64_t d = idx % p.dim;
  const int64_t slice = b * p.ngroups + d / (p.dim / p.ngroups);
  return {
      p.u + idx * L,
      p.delta + idx * L,
      p.z ? p.z + idx * L : nullptr,
      p.A + d * N,
      p.B + slice * L * N,
      p.C + slice * L * N,
      p.D ? p.D[d] : 0.f,
      p.delta_bias ? p.delta_bias[d] : 0.f,
  };
}

MAMBA_INLINE float row_dt(const SelectiveScanParams& p, const RowPtrs& r, int64_t t) {
  const float dt = r.delta[t] + r.bias;
  return p.delta_softplus ? softplus(dt) : dt;
}

// ---- forward ---------------------------------------------------------------

// h: scratch of size dstate.
MAMBA_INLINE void fwd_row_body(const SelectiveScanParams& p, int64_t idx, float* __restrict h) {
  const int64_t L = p.seqlen;
  const int64_t N = p.dstate;
  const RowPtrs r = row_ptrs(p, idx);
  const float* __restrict A = r.A;
  float* out = p.out + idx * L;

  for (int64_t n = 0; n < N; ++n) h[n] = 0.f;
  for (int64_t t = 0; t < L; ++t) {
    const float dt = row_dt(p, r, t);
    const float dt_u = dt * r.u[t];
    const float* __restrict Bt = r.B + t * N;
    const float* __restrict Ct = r.C + t * N;
    float y = 0.f;
    MAMBA_SIMD_REDUCE(y)
    for (int64_t n = 0; n < N; ++n) {
      const float hn = exp_approx(dt * A[n]) * h[n] + dt_u * Bt[n];
      h[n] = hn;
      y += hn * Ct[n];
    }
    y += r.D * r.u[t];
    if (r.z) y *= r.z[t] * sigmoid(r.z[t]);
    out[t] = y;
  }
  if (p.last_state) std::memcpy(p.last_state + idx * N, h, N * sizeof(float));
}

void fwd_row_default(const SelectiveScanParams& p, int64_t idx, float* h) {
  fwd_row_body(p, idx, h);
}

#if MAMBA_X86_DISPATCH
MAMBA_TARGET_AVX2 void fwd_row_avx2(const SelectiveScanParams& p, int64_t idx, float* h) {
  fwd_row_body(p, idx, h);
}
#endif

// ---- backward --------------------------------------------------------------

struct BwdScratch {
  float* hs;      // (seqlen, dstate): every state of the row
  float* dB_acc;  // (seqlen, dstate): this thread's partial dB for its current slice
  float* dC_acc;
  float* gh;      // (dstate)
  float* dA_acc;  // (dstate)
  const float* zeros;  // (dstate), stands in for h_{-1}
};

MAMBA_INLINE void bwd_row_body(const SelectiveScanBwdParams& p, int64_t idx, const BwdScratch& s) {
  const SelectiveScanParams& f = p.fwd;
  const int64_t L = f.seqlen;
  const int64_t N = f.dstate;
  const RowPtrs r = row_ptrs(f, idx);
  const float* __restrict A = r.A;
  const float* dout = p.dout + idx * L;
  float* du = p.du + idx * L;
  float* ddelta = p.ddelta + idx * L;
  float* dz = p.dz ? p.dz + idx * L : nullptr;
  float* __restrict gh = s.gh;
  float* __restrict dA_acc = s.dA_acc;

  // Forward recompute, keeping every state.
  for (int64_t t = 0; t < L; ++t) {
    const float dt = row_dt(f, r, t);
    const float dt_u = dt * r.u[t];
    const float* __restrict Bt = r.B + t * N;
    const float* __restrict h_prev = t > 0 ? s.hs + (t - 1) * N : s.zeros;
    float* __restrict h = s.hs + t * N;
    MAMBA_SIMD
    for (int64_t n = 0; n < N; ++n) h[n] = exp_approx(dt * A[n]) * h_prev[n] + dt_u * Bt[n];
  }

  // Reverse scan. gh holds dLoss/dh_t, carried back through h_t = a * h_{t-1} + dt*u*B.
  for (int64_t n = 0; n < N; ++n) {
    gh[n] = p.dlast_state ? p.dlast_state[idx * N + n] : 0.f;
    dA_acc[n] = 0.f;
  }
  float dD_acc = 0.f;
  float dbias_acc = 0.f;

  for (int64_t t = L - 1; t >= 0; --t) {
    const float dt = row_dt(f, r, t);
    const float u = r.u[t];
    const float dt_u = dt * u;
    const float* __restrict Bt = r.B + t * N;
    const float* __restrict Ct = r.C + t * N;
    const float* __restrict h = s.hs + t * N;
    const float* __restrict h_prev = t > 0 ? s.hs + (t - 1) * N : s.zeros;
    float* __restrict dBt = s.dB_acc + t * N;
    float* __restrict dCt = s.dC_acc + t * N;

    // out = y * silu(z), y = <h, C> + D*u
    float dy = dout[t];
    if (r.z) {
      float y = 0.f;
      MAMBA_SIMD_REDUCE(y)
      for (int64_t n = 0; n < N; ++n) y += h[n] * Ct[n];
      y += r.D * u;
      const float zt = r.z[t];
      const float sg = sigmoid(zt);
      dz[t] = dout[t] * y * sg * (1.f + zt * (1.f - sg));
      dy = dout[t] * zt * sg;
    }
    dD_acc += dy * u;

    float ddt = 0.f;
    float gh_dot_B = 0.f;
    MAMBA_SIMD_REDUCE(ddt, gh_dot_B)
    for (int64_t n = 0; n < N; ++n) {
      const float g = gh[n] + dy * Ct[n];
      dCt[n] += dy * h[n];
      const float a = exp_approx(dt * A[n]);
      const float g_a = g * h_prev[n] * a;  // dLoss/d(dt*A[n])
      dA_acc[n] += g_a * dt;
      ddt += g_a * A[n] + g * u * Bt[n];
      gh_dot_B += g * Bt[n];
      dBt[n] += g * dt_u;
      gh[n] = g * a;
    }
    du[t] = dy * r.D + dt * gh_dot_B;
    if (f.delta_softplus) ddt *= sigmoid(r.delta[t] + r.bias);
    ddelta[t] = ddt;
    dbias_acc += ddt;
  }

  std::memcpy(p.dA_part + idx * N, dA_acc, N * sizeof(float));
  p.dD_part[idx] = dD_acc;
  p.ddelta_bias_part[idx] = dbias_acc;
}

void bwd_row_default(const SelectiveScanBwdParams& p, int64_t idx, const BwdScratch& s) {
  bwd_row_body(p, idx, s);
}

#if MAMBA_X86_DISPATCH
MAMBA_TARGET_AVX2 void bwd_row_avx2(const SelectiveScanBwdParams& p, int64_t idx,
                                    const BwdScratch& s) {
  bwd_row_body(p, idx, s);
}
#endif

// ---- dispatch --------------------------------------------------------------

using FwdRowFn = void (*)(const SelectiveScanParams&, int64_t, float*);
using BwdRowFn = void (*)(const SelectiveScanBwdParams&, int64_t, const BwdScratch&);

struct Isa {
  const char* name;
  FwdRowFn fwd;
  BwdRowFn bwd;
};

const Isa kDefault{"default", fwd_row_default, bwd_row_default};
#if MAMBA_X86_DISPATCH
const Isa kAvx2{"avx2", fwd_row_avx2, bwd_row_avx2};
#endif

const Isa& best_isa() {
#if MAMBA_X86_DISPATCH
  __builtin_cpu_init();  // g_isa is initialized before libgcc's own cpu-init constructor may run
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return kAvx2;
#endif
  return kDefault;
}

const Isa* g_isa = &best_isa();

int clamp_threads(int num_threads, int64_t total_work, int64_t work_per_thread) {
  const int max_useful = static_cast<int>(std::max<int64_t>(1, total_work / work_per_thread));
  return std::min(num_threads, max_useful);
}

}  // namespace

const char* cpu_isa() { return g_isa->name; }

bool set_cpu_isa(const std::string& name) {
  if (name == "auto") {
    g_isa = &best_isa();
    return true;
  }
  if (name == kDefault.name) {
    g_isa = &kDefault;
    return true;
  }
#if MAMBA_X86_DISPATCH
  if (name == kAvx2.name && &best_isa() == &kAvx2) {
    g_isa = &kAvx2;
    return true;
  }
#endif
  return false;
}

void selective_scan_fwd_f32(const SelectiveScanParams& p, int num_threads) {
  const int64_t N = p.dstate;
  const FwdRowFn row = g_isa->fwd;
  // Don't spawn more threads than there is work for (decode-sized calls).
  num_threads = clamp_threads(num_threads, p.batch * p.dim * p.seqlen * N, 1 << 15);

  // Each (batch, channel) pair is an independent recurrence over time.
  parallel_for(0, p.batch * p.dim, num_threads, [&](int64_t lo, int64_t hi) {
    std::vector<float> h(N);
    for (int64_t idx = lo; idx < hi; ++idx) row(p, idx, h.data());
  });
}

void selective_scan_bwd_f32(const SelectiveScanBwdParams& p, int num_threads) {
  const SelectiveScanParams& f = p.fwd;
  const int64_t L = f.seqlen;
  const int64_t N = f.dstate;
  const int64_t dims_per_group = f.dim / f.ngroups;
  const BwdRowFn row = g_isa->bwd;
  num_threads = clamp_threads(num_threads, f.batch * f.dim * L * N, 1 << 14);

  // dB/dC are shared by every channel of a (batch, group) slice. Threads own
  // contiguous channel ranges, so each accumulates its current slice locally
  // and flushes it once under that slice's lock.
  std::vector<std::mutex> slice_locks(f.batch * f.ngroups);

  parallel_for(0, f.batch * f.dim, num_threads, [&](int64_t lo, int64_t hi) {
    std::vector<float> hs(L * N), dB_acc(L * N), dC_acc(L * N);
    std::vector<float> gh(N), dA_acc(N), zeros(N, 0.f);
    const BwdScratch scratch{hs.data(), dB_acc.data(), dC_acc.data(),
                             gh.data(), dA_acc.data(), zeros.data()};
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
      const int64_t slice = (idx / f.dim) * f.ngroups + (idx % f.dim) / dims_per_group;
      if (slice != cur_slice) {
        flush();
        cur_slice = slice;
        std::fill(dB_acc.begin(), dB_acc.end(), 0.f);
        std::fill(dC_acc.begin(), dC_acc.end(), 0.f);
      }
      row(p, idx, scratch);
    }
    flush();
  });
}

}  // namespace mamba_cpu
