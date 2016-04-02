cmake_minimum_required(VERSION 2.8)

include_directories(/usr/include/eigen3/)

file(GLOB SRC_LIST . *.cc)
add_library(core ${SRC_LIST})
target_link_libraries(core boost_program_options)
