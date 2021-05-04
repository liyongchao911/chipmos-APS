#set(LIB_SRC linked_list.cu job_base.cu machine_base.cu common.cpp chromosome_base.cu)
add_library(lib STATIC ${LIB_SRC})
target_compile_features(lib PUBLIC cxx_std_11)
set_target_properties(lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(lib PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

