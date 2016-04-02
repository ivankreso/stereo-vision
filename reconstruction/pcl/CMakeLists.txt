cmake_minimum_required(VERSION 2.8)

include_directories(/usr/include/eigen3/)

find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

file(GLOB SRC_LIST . *.cc)

#message(STATUS "${SRC_LIST}")

add_library(recon_base ${SRC_LIST})
