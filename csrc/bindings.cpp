#include <nanobind/nanobind.h>

#include <cstdint>
#include <thread>

#include "selective_scan.h"

namespace nb = nanobind;

// Tensors cross the boundary as raw data pointers (0 = None). The Python side
// guarantees dtype, contiguity and shapes; see mamba_cpu/ops.
template <typename T>
static T* as_ptr(uintptr_t p) {
  return reinterpret_cast<T*>(p);
}

NB_MODULE(_C, m) {
  m.doc() = "mamba_cpu native kernels";

  m.def(
      "selective_scan_fwd_f32",
      [](uintptr_t u, uintptr_t delta, uintptr_t A, uintptr_t B, uintptr_t C, uintptr_t D,
         uintptr_t z, uintptr_t delta_bias, uintptr_t out, uintptr_t last_state, int64_t batch,
         int64_t dim, int64_t seqlen, int64_t dstate, int64_t ngroups, bool delta_softplus,
         int num_threads) {
        mamba_cpu::SelectiveScanParams p{
            as_ptr<const float>(u),          as_ptr<const float>(delta),
            as_ptr<const float>(A),          as_ptr<const float>(B),
            as_ptr<const float>(C),          as_ptr<const float>(D),
            as_ptr<const float>(z),          as_ptr<const float>(delta_bias),
            as_ptr<float>(out),              as_ptr<float>(last_state),
            batch,                           dim,
            seqlen,                          dstate,
            ngroups,                         delta_softplus,
        };
        nb::gil_scoped_release release;
        mamba_cpu::selective_scan_fwd_f32(p, num_threads);
      },
      nb::arg("u"), nb::arg("delta"), nb::arg("A"), nb::arg("B"), nb::arg("C"), nb::arg("D"),
      nb::arg("z"), nb::arg("delta_bias"), nb::arg("out"), nb::arg("last_state"),
      nb::arg("batch"), nb::arg("dim"), nb::arg("seqlen"), nb::arg("dstate"),
      nb::arg("ngroups"), nb::arg("delta_softplus"), nb::arg("num_threads"));

  m.def(
      "selective_scan_bwd_f32",
      [](uintptr_t u, uintptr_t delta, uintptr_t A, uintptr_t B, uintptr_t C, uintptr_t D,
         uintptr_t z, uintptr_t delta_bias, uintptr_t dout, uintptr_t dlast_state, uintptr_t du,
         uintptr_t ddelta, uintptr_t dA_part, uintptr_t dB, uintptr_t dC, uintptr_t dD_part,
         uintptr_t dz, uintptr_t ddelta_bias_part, int64_t batch, int64_t dim, int64_t seqlen,
         int64_t dstate, int64_t ngroups, bool delta_softplus, int num_threads) {
        mamba_cpu::SelectiveScanBwdParams p{
            {
                as_ptr<const float>(u),   as_ptr<const float>(delta),
                as_ptr<const float>(A),   as_ptr<const float>(B),
                as_ptr<const float>(C),   as_ptr<const float>(D),
                as_ptr<const float>(z),   as_ptr<const float>(delta_bias),
                nullptr,                  nullptr,
                batch,                    dim,
                seqlen,                   dstate,
                ngroups,                  delta_softplus,
            },
            as_ptr<const float>(dout),
            as_ptr<const float>(dlast_state),
            as_ptr<float>(du),
            as_ptr<float>(ddelta),
            as_ptr<float>(dA_part),
            as_ptr<float>(dB),
            as_ptr<float>(dC),
            as_ptr<float>(dD_part),
            as_ptr<float>(dz),
            as_ptr<float>(ddelta_bias_part),
        };
        nb::gil_scoped_release release;
        mamba_cpu::selective_scan_bwd_f32(p, num_threads);
      },
      nb::arg("u"), nb::arg("delta"), nb::arg("A"), nb::arg("B"), nb::arg("C"), nb::arg("D"),
      nb::arg("z"), nb::arg("delta_bias"), nb::arg("dout"), nb::arg("dlast_state"),
      nb::arg("du"), nb::arg("ddelta"), nb::arg("dA_part"), nb::arg("dB"), nb::arg("dC"),
      nb::arg("dD_part"), nb::arg("dz"), nb::arg("ddelta_bias_part"), nb::arg("batch"),
      nb::arg("dim"), nb::arg("seqlen"), nb::arg("dstate"), nb::arg("ngroups"),
      nb::arg("delta_softplus"), nb::arg("num_threads"));
}
