cmake_minimum_required(VERSION 3.22)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 86 CACHE STRING "CUDA architectures" FORCE)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CXX_STANDARD 20)

enable_language(CUDA)
enable_testing()

include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

project(thrust_weekly CXX C CUDA)
file(GLOB_RECURSE thrust_weekly_srcs src/*.cc src/*.cu src/*.cuh src/*.cpp)

add_compile_definitions(THRUST_DEBUG_SYNC)

add_executable(thrust_weekly ${thrust_weekly_srcs})

target_link_directories(thrust_weekly PRIVATE /usr/local/cuda-11.6/lib64)
target_link_libraries(
        thrust_weekly
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(thrust_weekly)