cmake_minimum_required(VERSION 2.8)

file(GLOB SRC_LIST *.cc)
add_library(descriptor ${SRC_LIST})

