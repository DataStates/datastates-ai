SET(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "disable tests for googlebench")
SET(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "disable installation for googlebench")
SET(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "disable docs installation for googlebench")
FetchContent_Declare(
  googlebench
  GIT_REPOSITORY https://github.com/google/benchmark
  GIT_TAG        344117638c8ff7e239044fd0fa7085839fc03021 # release-1.8.3
)
FetchContent_MakeAvailable(googlebench)

add_executable(server_benchmarks server_benchmarks.cc)
target_link_libraries(server_benchmarks PRIVATE benchmark::benchmark_main)
