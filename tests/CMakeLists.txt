SET(INSTALL_GTEST OFF CACHE BOOL "install the bundeled version of gtest")
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        f8d7d77c06936315286eb55f8de22cd23c188571 # release-1.14.0
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

add_executable(dstatesai_server_tests dstatesai_server_tests.cc)
target_link_libraries(dstatesai_server_tests PRIVATE dstatesai_server GTest::gtest_main)
gtest_discover_tests(dstatesai_server_tests)
