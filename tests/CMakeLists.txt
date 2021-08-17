
set(CMAKE_CXX_STANDARD 11)
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

file(GLOB_RECURSE cuda_test_cpp_sources ${CMAKE_CURRENT_SOURCE_DIR}/cuda_test_*/test_*.cpp)
file(GLOB_RECURSE cuda_test_cu_sources ${CMAKE_CURRENT_SOURCE_DIR}/cuda_test_*/*.cu)


file(GLOB_RECURSE test_cpp_sources ${CMAKE_CURRENT_SOURCE_DIR}/test_*/test_*.cpp)

add_executable(
    test
    main.cpp
    ${test_cpp_sources}
)

if (WIN32)
    target_link_libraries(
        test
        src
        pgalib
        -static-libstdc++
        m
        gtest_main
    )
else()
    target_link_libraries(
        test
        src
        pgalib
        stdc++
        m
        gtest_main
    )
endif()

include(GoogleTest)
gtest_discover_tests(test)
