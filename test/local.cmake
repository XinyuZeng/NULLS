add_executable(skewnull test/test.cpp)
target_link_libraries(skewnull fmt arrow_shared)

include(GoogleTest)
file(GLOB_RECURSE TEST_SRC "test/*_test.cpp")
foreach (TEST_FILE ${TEST_SRC})
    cmake_path(GET TEST_FILE STEM TEST_EXE)
    add_executable(${TEST_EXE} ${TEST_FILE})
    target_link_libraries(
            ${TEST_EXE}
            GTest::gtest_main
            GTest::gmock
            ${MEM_LIB}
    )
    target_compile_options(${TEST_EXE} PRIVATE -mavx512f -mavx512bitalg -funroll-loops -march=native)
    #  -Rpass-analysis=loop-vectorize
    gtest_discover_tests(${TEST_EXE})
endforeach ()
