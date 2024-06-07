include(FetchContent)

FetchContent_Declare(
  xsimd
  GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
  GIT_TAG 12.1.1
  GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(xsimd)

FetchContent_GetProperties(xsimd)