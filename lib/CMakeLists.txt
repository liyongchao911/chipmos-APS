set(LIB_SRC linked_list.c job_base.c machine_base.c common.c chromosome_base.c)
add_library(lib STATIC ${LIB_SRC})
target_include_directories(lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
