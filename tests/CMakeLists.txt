
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

file(GLOB_RECURSE test_cpp_sources ${CMAKE_CURRENT_SOURCE_DIR}/test_*/test_*.cpp)

set(TEST_SRC
        test_entities/test_entities_ctor.cpp
        test_entity/test_entity.cpp
        test_lot/test_lot.cpp
        test_lots/test_lots.cpp
        test_machines/test_machines_addMachine.cpp
        test_machines/test_machines_addPrescheduledJob.cpp 
        test_machines/test_machines_prescheduleJobs.cpp
        test_machines/test_machines_scheduleAGroup.cpp
        test_machines/test_machines_addGroupJobs.cpp
        test_algorithm/test_prescheduling.cpp
        test_machines/test_machines_groupBy.cpp
        test_machines/test_machines_distributeAResource.cpp
        test_machines/test_machines_canJobRunOnTheMachine.cpp
        test_machines/test_machines_addNewResource.cpp
        test_csv/test_csv_read.cpp
        test_csv/test_csv_parse.cpp
        )

add_executable(
    test
    main.cpp
    ${TEST_SRC}
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
