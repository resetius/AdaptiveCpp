// TEMPORARY: every work item does a 64-bit fetch_add on its own element, so the
// lanes of a SIMD group hit different addresses and aggregation must not apply.
#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
  const std::size_t n = argc > 1 ? std::atoll(argv[1]) : 2048;
  const std::size_t rounds = argc > 2 ? std::atoll(argv[2]) : 4;

  sycl::queue q;
  auto *data = sycl::malloc_shared<unsigned long long>(n, q);
  for (std::size_t i = 0; i < n; ++i) {
    data[i] = 0;
  }

  for (std::size_t r = 0; r < rounds; ++r) {
    q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx) {
       sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed,
                        sycl::memory_scope::device>
           c{data[idx[0]]};
       c.fetch_add(idx[0] + 1);
     }).wait();
  }

  std::size_t wrong = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (data[i] != rounds * (i + 1)) {
      ++wrong;
    }
  }
  std::printf("%zu items, %zu rounds: %zu wrong (data[0]=%llu data[%zu]=%llu) %s\n", n,
              rounds, wrong, data[0], n - 1, data[n - 1], wrong == 0 ? "OK" : "WRONG");
  sycl::free(data, q);
  return wrong == 0 ? 0 : 1;
}
