function(add_benchmark TARGET_NAME LIBS COMP_OPTIONS)
    set(BENCH_TARGET ${TARGET_NAME})
    add_executable(${BENCH_TARGET} bench/${BENCH_TARGET}.cpp)
    target_link_libraries(${BENCH_TARGET} PRIVATE ${LIBS})
    if (COMP_OPTIONS)
        check_cxx_compiler_flag(${COMP_OPTIONS} HAS_FLAG)
        if(HAS_FLAG)
        else()
        message(STATUS "The flag ${FLAG} is not supported by the current compiler")
        endif()
    endif()
    target_compile_options(${BENCH_TARGET} PRIVATE ${COMP_OPTIONS})
endfunction()

add_benchmark("bench_arrow_cp" "benchmark::benchmark;arrow_shared;parquet_shared;arrow_test" "-mbmi2")
add_benchmark("bench_avx_agg" "benchmark::benchmark;fmt" "-mavx512bw")
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set(ASAN "-fsanitize=address")
endif()
add_benchmark("bench_in_mem" "benchmark::benchmark;${MEM_LIB}" "${ASAN}")
add_benchmark("bench_dense_to_spaced" "benchmark::benchmark;${MEM_LIB};roaring" 
"-mavx512f;-mavx512bitalg;-funroll-loops;-march=native;-msse4.2;-mavx512bw;-Ofast")

add_benchmark("motivation" "benchmark::benchmark;arrow_shared;parquet_shared;arrow_test" "")

add_benchmark("read_pq" "arrow_shared;parquet_shared" "")
add_benchmark("bench_vec_load" "benchmark::benchmark" "-mavx512f;-mavx512bw;-mavx512bitalg;-march=native;${ASAN}")

add_benchmark("roaring_cr" "roaring" "-mbmi2;-mavx512bitalg;-funroll-loops;-march=native")
