cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")

file(GLOB SRC_LIST *.cc)
set(SRC_LIST ${SRC_LIST})

message(STATUS "${SRC_LIST}")

add_executable(vo_eval ${SRC_LIST})
