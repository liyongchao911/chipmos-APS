find_package(GTest)
find_package(CUDA REQUIRED)
include_directories(${GTest_INCLUDE_DIRS})
file(GLOB_RECURSE cppsources ${CMAKE_CURRENT_SOURCE_DIR}/*/test_*.cpp)
file(GLOB_RECURSE cusources ${CMAKE_CURRENT_SOURCE_DIR}/*/*.cu)
add_executable(test main.cpp ${cppsources} ${cusources})
set_property(TARGET test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test lib ${GTest_LIBS} gtest gtest_main Threads::Threads ${CUDA_LIBRARIES})