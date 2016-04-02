# run example: (in build dir)
# in Debug/ folder:
# cmake src_dir -DCMAKE_BUILD_TYPE=debug
# in Release/ folder:
# cmake src_dir -DCMAKE_BUILD_TYPE=release

cmake_minimum_required(VERSION 2.8)

file(GLOB SRC_LIST . *.cc)
add_subdirectory(../mono/ libs/mono)
add_subdirectory(../refiner/ libs/refiner)

if(NOT TARGET tracker_base)
  add_subdirectory(../base/ libs/base)
endif()
if(NOT TARGET core)
  add_subdirectory(../../core/ libs/core)
endif()
#if(NOT TARGET recon_base)
#  add_subdirectory(../../reconstruction/base/ libs/recon_base)
#endif()

add_library(tracker_stereo ${SRC_LIST})
#target_link_libraries(tracker_stereo core tracker_base tracker_mono tracker_refiner recon_base)
target_link_libraries(tracker_stereo core tracker_base tracker_mono tracker_refiner)

