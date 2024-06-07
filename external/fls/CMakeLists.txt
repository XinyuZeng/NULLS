add_library(x86_64_avx512bw_intrinsic_1024_uf1_rsum OBJECT
            x86_64_avx512bw_intrinsic_1024_uf1_rsum_src.cpp)
target_compile_definitions(x86_64_avx512bw_intrinsic_1024_uf1_rsum PRIVATE IS_SCALAR)
set(FLAG -mavx512bw -fno-builtin)
check_cxx_compiler_flag(${FLAG} HAS_FLAG)
if(HAS_FLAG)
else()
 message(STATUS "The flag ${FLAG} is not supported by the current compiler")
endif()
target_compile_options(x86_64_avx512bw_intrinsic_1024_uf1_rsum PUBLIC ${FLAG})
include(CMakePrintHelpers)
cmake_print_properties(TARGETS x86_64_avx512bw_intrinsic_1024_uf1_rsum
                       PROPERTIES COMPILE_DEFINITIONS
                       PROPERTIES COMPILE_OPTIONS)

add_library(x86_64_avx512bw_intrinsic_1024_uf1_unpack OBJECT
x86_64_avx512bw_intrinsic_1024_uf1_unpack_src.cpp)
target_compile_definitions(x86_64_avx512bw_intrinsic_1024_uf1_unpack PRIVATE IS_SCALAR)
target_compile_options(x86_64_avx512bw_intrinsic_1024_uf1_unpack PUBLIC ${FLAG})
cmake_print_properties(TARGETS x86_64_avx512bw_intrinsic_1024_uf1_unpack
                       PROPERTIES COMPILE_DEFINITIONS
                       PROPERTIES COMPILE_OPTIONS)

add_library(helper_scalar_av_1024_uf1_pack_src helper_scalar_av_1024_uf1_pack_src.cpp)