function(build_arrow)
    set(one_value_args)
    set(multi_value_args)

    cmake_parse_arguments(ARG
            "${options}"
            "${one_value_args}"
            "${multi_value_args}"
            ${ARGN})
    if (ARG_UNPARSED_ARGUMENTS)
        message(SEND_ERROR "Error: unrecognized arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif ()

    # If Arrow needs to be built, the default location will be within the build tree.
    set(ARROW_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/arrow_ep-prefix")

    set(ARROW_SHARED_LIBRARY_DIR "${ARROW_PREFIX}/lib")

    set(ARROW_SHARED_LIB_FILENAME
            "${CMAKE_SHARED_LIBRARY_PREFIX}arrow${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(ARROW_SHARED_LIB "${ARROW_SHARED_LIBRARY_DIR}/${ARROW_SHARED_LIB_FILENAME}")
    set(PARQUET_SHARED_LIB_FILENAME
            "${CMAKE_SHARED_LIBRARY_PREFIX}parquet${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(PARQUET_SHARED_LIB "${ARROW_SHARED_LIBRARY_DIR}/${PARQUET_SHARED_LIB_FILENAME}")
    set(ARROW_TEST_SHARED_LIB_FILENAME
            "${CMAKE_SHARED_LIBRARY_PREFIX}arrow_testing${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(ARROW_TEST_SHARED_LIB "${ARROW_SHARED_LIBRARY_DIR}/${ARROW_TEST_SHARED_LIB_FILENAME}")

    set(ARROW_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/arrow_ep-build")
    #     set(ARROW_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${ARROW_PREFIX}"
    #             "-DCMAKE_INSTALL_LIBDIR=lib" "-Dxsimd_SOURCE=BUNDLED"
    #             "-DARROW_BUILD_STATIC=OFF" "-DARROW_PARQUET=ON"
    #             "-DARROW_WITH_UTF8PROC=OFF" "-DARROW_WITH_RE2=OFF"
    #             "-DARROW_FILESYSTEM=ON" "-DARROW_CSV=ON" "-DARROW_PYTHON=ON"
    #             "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
    set(ARROW_CMAKE_ARGS "--preset ninja-release" 
    "-DCMAKE_INSTALL_PREFIX=${ARROW_PREFIX}" "-DARROW_DEPENDENCY_SOURCE=AUTO"
    "-DCMAKE_INSTALL_LIBDIR=lib" "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON" "-DCMAKE_INSTALL_RPATH=${ARROW_PREFIX}/lib"
    "-DARROW_CUDA=OFF" "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}" "-DARROW_BUILD_BENCHMARKS=ON" "-DARROW_BUILD_BENCHMARKS_REFERENCE=ON")
            
    set(ARROW_INCLUDE_DIR "${ARROW_PREFIX}/include")
    # xinyu: make sure parent have access to this variable
    set(ARROW_INCLUDE_DIR "${ARROW_PREFIX}/include" PARENT_SCOPE)

    set(ARROW_BUILD_BYPRODUCTS "${ARROW_SHARED_LIB}" "${PARQUET_SHARED_LIB}" "${ARROW_TEST_SHARED_LIB}")

    include(ExternalProject)
    cmake_policy(SET CMP0135 NEW)
    externalproject_add(arrow_ep
            URL https://github.com/apache/arrow/archive/refs/tags/apache-arrow-14.0.0.tar.gz
            SOURCE_SUBDIR cpp
            BINARY_DIR "${ARROW_BINARY_DIR}"
            CMAKE_ARGS "${ARROW_CMAKE_ARGS}"
            BUILD_BYPRODUCTS "${ARROW_BUILD_BYPRODUCTS}")

    set(ARROW_LIBRARY_TARGET arrow_shared)
    set(PARQUET_LIBRARY_TARGET parquet_shared)
    set(ARROW_TEST_LIBRARY_TARGET arrow_test)


    file(MAKE_DIRECTORY "${ARROW_INCLUDE_DIR}")
    add_library(${ARROW_LIBRARY_TARGET} SHARED IMPORTED)
    add_library(${PARQUET_LIBRARY_TARGET} SHARED IMPORTED)
    add_library(${ARROW_TEST_LIBRARY_TARGET} SHARED IMPORTED)
    set_target_properties(${ARROW_LIBRARY_TARGET}
            PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${ARROW_INCLUDE_DIR}
            IMPORTED_LOCATION ${ARROW_SHARED_LIB})
    set_target_properties(${PARQUET_LIBRARY_TARGET}
            PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${ARROW_INCLUDE_DIR}
            IMPORTED_LOCATION ${PARQUET_SHARED_LIB})
    set_target_properties(${ARROW_TEST_LIBRARY_TARGET}
            PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${ARROW_INCLUDE_DIR}
            IMPORTED_LOCATION ${ARROW_TEST_SHARED_LIB})

    add_dependencies(${ARROW_LIBRARY_TARGET} arrow_ep)
endfunction()
build_arrow()