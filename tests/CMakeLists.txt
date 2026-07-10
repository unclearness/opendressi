include(FetchContent)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# CPU-only tests (always runnable)
add_executable(dressi_tests_cpu
    test_graph.cpp
    test_infer.cpp
    test_cpu_eval.cpp
    test_backward.cpp
    test_packing.cpp
    test_codegen.cpp)
target_link_libraries(dressi_tests_cpu PRIVATE dressi GTest::gtest_main)
target_include_directories(dressi_tests_cpu PRIVATE ${CMAKE_SOURCE_DIR}/src)
if(MSVC)
    target_compile_options(dressi_tests_cpu PRIVATE /W4 /permissive- /bigobj)
endif()
gtest_discover_tests(dressi_tests_cpu PROPERTIES LABELS cpu)

# GPU-dependent tests (require a Vulkan device; excluded via `ctest -LE gpu`)
add_executable(test_vk_smoke test_vk_smoke.cpp)
target_link_libraries(test_vk_smoke PRIVATE dressi_vkw GTest::gtest_main)
gtest_discover_tests(test_vk_smoke PROPERTIES LABELS gpu)

add_executable(dressi_tests_gpu
    test_executor_naive.cpp
    test_executor_rsp.cpp
    test_reactive.cpp
    test_raster.cpp)
target_link_libraries(dressi_tests_gpu PRIVATE dressi GTest::gtest_main)
target_include_directories(dressi_tests_gpu PRIVATE ${CMAKE_SOURCE_DIR}/src)
if(MSVC)
    target_compile_options(dressi_tests_gpu PRIVATE /W4 /permissive- /bigobj)
endif()
gtest_discover_tests(dressi_tests_gpu PROPERTIES LABELS gpu)
