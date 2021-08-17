cmake_minimum_required(VERSION 3.8)
project(ChipmosWb LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
include_directories(${CMAKE_SOURCE_DIR})
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(
    main 
    main.cpp 
)

add_subdirectory(lib)

add_subdirectory(src)
add_subdirectory(tests)

if(WIN32)
    target_link_libraries(main pgalib src -static-libstdc++ m)
else()
    target_link_libraries(main pgalib src stdc++ m)
endif()
