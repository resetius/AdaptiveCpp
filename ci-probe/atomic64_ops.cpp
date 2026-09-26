// TEMPORARY: one 64-bit atomic operation per run, so that a device which fails
// to build a kernel names the operation.
#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <string_view>

template <class T, class Op>
static bool run(sycl::queue &q, std::size_t n, const char *name, T init, Op op) {
  auto *p = sycl::malloc_shared<T>(1, q);
  *p = init;
  q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx) {
     sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> r{*p};
     op(r, (T)(idx[0] + 1));
   }).wait();
  std::printf("%-12s result = %llu\n", name, (unsigned long long)*p);
  sycl::free(p, q);
  return true;
}

int main(int argc, char **argv) {
  const std::string_view what = argc > 1 ? argv[1] : "add";
  const std::size_t n = argc > 2 ? std::atoll(argv[2]) : 2048;

  sycl::queue q;
  using u64 = unsigned long long;
  using i64 = long long;

  if (what == "add")      return run<u64>(q, n, "add", 0, [](auto &r, u64 x) { r.fetch_add(x); }) ? 0 : 1;
  if (what == "add_used") return run<u64>(q, n, "add_used", 0, [](auto &r, u64 x) { volatile u64 s = r.fetch_add(x); (void)s; }) ? 0 : 1;
  if (what == "sub")      return run<u64>(q, n, "sub", ~0ull, [](auto &r, u64 x) { r.fetch_sub(x); }) ? 0 : 1;
  if (what == "add_i64")  return run<i64>(q, n, "add_i64", 0, [](auto &r, i64 x) { r.fetch_add(x); }) ? 0 : 1;
  if (what == "sub_i64")  return run<i64>(q, n, "sub_i64", 0, [](auto &r, i64 x) { r.fetch_sub(x); }) ? 0 : 1;
  if (what == "exchange") return run<u64>(q, n, "exchange", 0, [](auto &r, u64 x) { r.exchange(x); }) ? 0 : 1;
  if (what == "load")     return run<u64>(q, n, "load", 7, [](auto &r, u64) { volatile u64 v = r.load(); (void)v; }) ? 0 : 1;
  if (what == "store")    return run<u64>(q, n, "store", 0, [](auto &r, u64 x) { r.store(x); }) ? 0 : 1;
  if (what == "and")      return run<u64>(q, n, "and", ~0ull, [](auto &r, u64 x) { r.fetch_and(x); }) ? 0 : 1;
  if (what == "or")       return run<u64>(q, n, "or", 0, [](auto &r, u64 x) { r.fetch_or(x); }) ? 0 : 1;
  if (what == "xor")      return run<u64>(q, n, "xor", 0, [](auto &r, u64 x) { r.fetch_xor(x); }) ? 0 : 1;
  if (what == "min")      return run<u64>(q, n, "min", ~0ull, [](auto &r, u64 x) { r.fetch_min(x); }) ? 0 : 1;
  if (what == "max")      return run<u64>(q, n, "max", 0, [](auto &r, u64 x) { r.fetch_max(x); }) ? 0 : 1;
  if (what == "cas")      return run<u64>(q, n, "cas", 0, [](auto &r, u64 x) { u64 e = 0; r.compare_exchange_strong(e, x); }) ? 0 : 1;

  if (what == "min_i64")  return run<i64>(q, n, "min_i64", (i64)1 << 40, [](auto &r, i64 x) { r.fetch_min(x); }) ? 0 : 1;
  if (what == "max_i64")  return run<i64>(q, n, "max_i64", -1, [](auto &r, i64 x) { r.fetch_max(x); }) ? 0 : 1;
  if (what == "and_i64")  return run<i64>(q, n, "and_i64", -1, [](auto &r, i64 x) { r.fetch_and(x); }) ? 0 : 1;
  if (what == "or_i64")   return run<i64>(q, n, "or_i64", 0, [](auto &r, i64 x) { r.fetch_or(x); }) ? 0 : 1;
  if (what == "xor_i64")  return run<i64>(q, n, "xor_i64", 0, [](auto &r, i64 x) { r.fetch_xor(x); }) ? 0 : 1;
  if (what == "load_i64") return run<i64>(q, n, "load_i64", 7, [](auto &r, i64) { volatile i64 v = r.load(); (void)v; }) ? 0 : 1;
  if (what == "exch_i64") return run<i64>(q, n, "exch_i64", 0, [](auto &r, i64 x) { r.exchange(x); }) ? 0 : 1;
  if (what == "cas_i64")  return run<i64>(q, n, "cas_i64", 0, [](auto &r, i64 x) { i64 e = 0; r.compare_exchange_strong(e, x); }) ? 0 : 1;
  if (what == "ptr_add") {
    // a pointer atomic, as the test does it
    int *base = sycl::malloc_shared<int>(4, q);
    auto *slot = sycl::malloc_shared<int *>(1, q);
    *slot = base;
    q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> idx) {
       sycl::atomic_ref<int *, sycl::memory_order::relaxed, sycl::memory_scope::device> r{*slot};
       r.fetch_add((std::ptrdiff_t)0);
     }).wait();
    std::printf("%-12s result = %s\n", "ptr_add", *slot == base ? "base" : "moved");
    sycl::free(slot, q);
    sycl::free(base, q);
    return 0;
  }

  std::printf("unknown operation: %s\n", argv[1]);
  return 2;
}
