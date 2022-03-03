# file(GLOB_RECURSE cpp_src_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
# file(GLOB_RECURSE c_src_files ${CMAKE_CURRENT_SOURCE_DIR}/*.c)
set(src_files
        algorithm.cpp
        chromosome.cpp
        condition_card.cpp
        csv.cpp
        da.cpp
        entities.cpp
        entity.cpp
        info.c
        infra.cpp
        job.cpp
        lot.cpp
        lots.cpp
        machines.cpp
        machine.cpp
        route.cpp
        record_gap.cpp
        machine_constraint.cpp
        machine_constraint_r.cpp
        machine_constraint_a.cpp
        arg_parser.cpp
        system_log.cpp
        time_converter.cpp
#         progress.c
        )
set(CMAKE_BUILD_TYPE Debug)
add_library(src STATIC ${src_files})
target_compile_features(src PUBLIC cxx_std_17)
target_include_directories(src PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
