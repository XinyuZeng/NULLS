# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

file(GLOB_RECURSE SRC_CPP mem/*.cpp)
file(GLOB_RECURSE INCLUDE_HPP mem/*.hpp mem/*.h)
# list(REMOVE_ITEM SRC_CPP ${CMAKE_CURRENT_SOURCE_DIR}/mem/main.cpp)
list(REMOVE_ITEM SRC_CPP "${CMAKE_CURRENT_SOURCE_DIR}/mem/utils/spaced_expand_fused.cpp")
list(REMOVE_ITEM SRC_CPP "${CMAKE_CURRENT_SOURCE_DIR}/mem/utils/sse4_bit_util.cpp")
list(REMOVE_ITEM SRC_CPP "${CMAKE_CURRENT_SOURCE_DIR}/mem/utils/avx2_bit_util.cpp")

set(FLAG -mavx2 -mavx512f -mavx512bitalg -funroll-loops -march=native)
set(SSE4FLAG -msse4.2 -mbmi2 -mpopcnt -funroll-loops -march=native)
set(AVX2FLAG -msse4.2 -mavx2 -mbmi2 -mpopcnt -funroll-loops -march=native)
# set(FLAG -mavx512bw -funroll-loops -march=native -Rpass-analysis=loop-vectorize -ftree-vectorize -mavx2)
check_cxx_compiler_flag(${FLAG} HAS_FLAG)
if(HAS_FLAG)
else()
 message(STATUS "The flag ${FLAG} is not supported by the current compiler")
endif()

set(MEM_LIB "mem_test_lib")

add_library(${MEM_LIB}_trunk ${SRC_CPP} ${INCLUDE_HPP})
target_link_libraries(${MEM_LIB}_trunk PUBLIC fmt ${ARROW_LIBS} xsimd)
target_include_directories(${MEM_LIB}_trunk PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/mem)
target_compile_options(${MEM_LIB}_trunk PRIVATE ${FLAG})

add_library(${MEM_LIB}_SSE4_lib mem/utils/sse4_bit_util.cpp)
target_link_libraries(${MEM_LIB}_SSE4_lib PUBLIC fmt ${ARROW_LIBS} xsimd)
target_include_directories(${MEM_LIB}_SSE4_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/mem)
target_compile_options(${MEM_LIB}_SSE4_lib PRIVATE ${SSE4FLAG})

add_library(${MEM_LIB}_AVX2_lib mem/utils/avx2_bit_util.cpp)
target_link_libraries(${MEM_LIB}_AVX2_lib PUBLIC fmt ${ARROW_LIBS} xsimd)
target_include_directories(${MEM_LIB}_AVX2_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/mem)
target_compile_options(${MEM_LIB}_AVX2_lib PRIVATE ${AVX2FLAG})

add_library(${MEM_LIB} INTERFACE)
target_link_libraries(${MEM_LIB} INTERFACE ${MEM_LIB}_trunk ${MEM_LIB}_SSE4_lib ${MEM_LIB}_AVX2_lib)
target_include_directories(${MEM_LIB} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/mem)

