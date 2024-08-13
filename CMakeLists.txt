cmake_minimum_required (VERSION 3.25)
project (datastates_ai VERSION 0.1.0
  DESCRIPTION "capture states of data for AI models"
  HOMEPAGE_URL "https://github.com/datastates/dstates_ai"
  LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-D__BENCHMARK -D__ASSERT -D__INFO)

if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE Release
    CACHE STRING "Choose the type of build" FORCE)
  set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif ()

option(DSTATES_AI_BUILD_EXAMPLES "Build examples" ON)
option(DSTATES_AI_BUILD_BENCHMARKS "Build microbenchmarks" ON)
option(DSTATES_AI_BUILD_TESTS "Build unit-tests" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)
find_package(thallium REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(nanobind REQUIRED HINTS ${Python_SITELIB}/nanobind/cmake)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit)

include_directories(BEFORE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src/common)
add_subdirectory(src)
