#pragma once

#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

namespace mamba_cpu {

// Splits [begin, end) into up to `num_threads` contiguous chunks and runs
// f(chunk_begin, chunk_end) on each, the last chunk on the calling thread.
// Threads are spawned per call; a persistent pool is a later optimization.
template <typename F>
inline void parallel_for(int64_t begin, int64_t end, int num_threads, const F& f) {
  const int64_t n = end - begin;
  if (n <= 0) return;
  const int64_t nt = std::min<int64_t>(std::max(num_threads, 1), n);
  if (nt == 1) {
    f(begin, end);
    return;
  }
  std::vector<std::thread> workers;
  workers.reserve(nt - 1);
  const int64_t chunk = n / nt;
  const int64_t rem = n % nt;
  int64_t start = begin;
  for (int64_t i = 0; i < nt; ++i) {
    const int64_t s = start;
    const int64_t e = s + chunk + (i < rem ? 1 : 0);
    start = e;
    if (i == nt - 1) {
      f(s, e);
    } else {
      workers.emplace_back([&f, s, e] { f(s, e); });
    }
  }
  for (auto& w : workers) w.join();
}

}  // namespace mamba_cpu
