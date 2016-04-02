# run example: (in build dir)
# in Debug/ folder:
# cmake src_dir -DCMAKE_BUILD_TYPE=debug
# in Release/ folder:
# cmake src_dir -DCMAKE_BUILD_TYPE=release

cmake_minimum_required(VERSION 2.8)

file(GLOB_RECURSE SRC_LIST *.cc *.c)
list(REMOVE_ITEM SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/birch/convolveFloatSSE_w32.c ${CMAKE_CURRENT_SOURCE_DIR}/birch/convolve_w32.c)

add_subdirectory(../../core/ libs/core)
add_subdirectory(../base/ libs/base)
add_subdirectory(../detector/ libs/detector)

message(STATUS "${SRC_LIST}")
message(STATUS "${OpenCV_LIBS}")

add_library(tracker_mono ${SRC_LIST})
target_link_libraries(tracker_mono core tracker_base detector)
