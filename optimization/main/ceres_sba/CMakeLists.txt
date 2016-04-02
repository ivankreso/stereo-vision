cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

include_directories(/usr/include/eigen3/)

add_subdirectory(../../sba libs/sba)

file(GLOB SRC_LIST *.cc *.c)
message(STATUS "${SRC_LIST}")

add_executable(ceres_test ${SRC_LIST})
target_link_libraries(ceres_test optim_sba)

