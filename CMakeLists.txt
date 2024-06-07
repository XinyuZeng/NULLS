cmake_minimum_required(VERSION 3.27)
set (CMAKE_CXX_STANDARD 20)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()
MESSAGE( STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE} )

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external")

project(NullsRevisited)
set(PROJECT_VERSION "0.0.1")

add_subdirectory(external/FastPFor)
add_subdirectory(external/lz4/build/cmake)
set(BENCHMARK_ENABLE_LIBPFM ON)
add_subdirectory(external/benchmark)
# Arrow
set(CMAKE_VENDOR_DIR "${CMAKE_SOURCE_DIR}/cmake/")
include("${CMAKE_VENDOR_DIR}/arrow.cmake")
include_directories("${ARROW_INCLUDE_DIR}")
set(ARROW_LIBS arrow_test arrow_shared parquet_shared)
message(STATUS "ARROW_LIBS: ${ARROW_LIBS}")
include("${CMAKE_VENDOR_DIR}/croaring.cmake")
include_directories(external/fls)
add_subdirectory(external/fls)
include("${CMAKE_VENDOR_DIR}/xsimd.cmake")

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set (CMAKE_CXX_FLAGS_RELEASE "-Wall -Wcast-align -Ofast -lm -DNDEBUG -std=c++20 -funroll-loops -msse4.2 -mpopcnt -mbmi2")
    set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-Wall -ggdb -Ofast -lm -DNDEBUG -std=c++20 -funroll-loops -msse4.2 -mpopcnt -mbmi2")
    set (CMAKE_CXX_FLAGS_DEBUG   "-Wall -Wcast-align -Wextra -ggdb -lm -DDEBUG -std=c++20 -msse4.2")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang")
    set (CMAKE_CXX_FLAGS_RELEASE "-Wall -Wcast-align -O3 -DNDEBUG -std=c++20 -funroll-loops -msse4.2 -mpopcnt -mbmi2 -Rpass-analysis=loop-vectorize")
    set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-Wall -Wcast-align -O3 -ggdb -DNDEBUG -std=c++20 -funroll-loops -msse4.2 -mpopcnt -mbmi2 -Rpass-analysis=loop-vectorize")
    set (CMAKE_CXX_FLAGS_DEBUG   "-Wall -Wcast-align -Wextra -ggdb -DDEBUG -std=c++20 -msse4.2 -mpopcnt -mbmi2")
else ()
    message(FATAL_ERROR "Please use GCC or Clang")
endif()

MESSAGE(STATUS "CMAKE_CXX_FLAGS_DEBUG: " ${CMAKE_CXX_FLAGS_DEBUG} )
MESSAGE(STATUS "CMAKE_CXX_FLAGS_RELEASE: " ${CMAKE_CXX_FLAGS_RELEASE} )

include_directories(${PROJECT_NAME}_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src 
    ${CMAKE_CURRENT_SOURCE_DIR}/external/FastPFor/headers
    ${CMAKE_CURRENT_SOURCE_DIR}/external/lz4/lib
    ${roaring_SOURCE_DIR}/include/)
file(GLOB_RECURSE SRC_CPP src/*.cpp)
# Separate lib for D->S convertion
get_filename_component(BIT_UTIL_CPP ${CMAKE_CURRENT_SOURCE_DIR}/src/bit_util.cpp ABSOLUTE)
list(REMOVE_ITEM SRC_CPP "${BIT_UTIL_CPP}")
add_library(D_S_CONVERSION_lib "${BIT_UTIL_CPP}")
target_link_libraries(D_S_CONVERSION_lib xsimd)
add_library(${PROJECT_NAME}_lib ${SRC_CPP})
target_compile_options(${PROJECT_NAME}_lib PRIVATE -march=native -mavx512bw)
target_link_libraries(${PROJECT_NAME}_lib arrow_shared x86_64_avx512bw_intrinsic_1024_uf1_rsum 
fmt 
helper_scalar_av_1024_uf1_pack_src x86_64_avx512bw_intrinsic_1024_uf1_unpack
xsimd D_S_CONVERSION_lib)
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_lib FastPFOR lz4 arrow_shared parquet_shared benchmark::benchmark roaring)
target_compile_options(${PROJECT_NAME} PRIVATE -march=native -mavx512bw)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(ASAN "-fsanitize=address")
    target_compile_options(${PROJECT_NAME} PRIVATE ${ASAN}  -fno-omit-frame-pointer)
    target_link_options(${PROJECT_NAME} PRIVATE ${ASAN})
endif()

include(mem/local.cmake)
include_directories(mem)
include(test/local.cmake)
include(bench/local.cmake)
