// TEMPORARY: 64-bit fetch_add of many work items on one address, the case that
// hangs the paravirtual GPU.
#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
  const std::size_t n = argc > 1 ? std::atoll(argv[1]) : 2048;

  sycl::queue q;
  std::printf("device: %s, atomic64 aspect: %d\n",
              q.get_device().get_info<sycl::info::device::name>().c_str(),
              (int)q.get_device().has(sycl::aspect::atomic64));
  std::fflush(stdout);

  auto *counter = sycl::malloc_shared<unsigned long long>(1, q);
  *counter = 0;

  std::printf("%zu work items on one address\n", n);
  std::fflush(stdout);

  q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1>) {
     sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed,
                      sycl::memory_scope::device>
         c{*counter};
     c.fetch_add(1ull);
   }).wait();

  const bool ok = *counter == n;
  std::printf("counter = %llu (expected %zu) %s\n", *counter, n, ok ? "OK" : "WRONG");
  sycl::free(counter, q);
  return ok ? 0 : 1;
}
