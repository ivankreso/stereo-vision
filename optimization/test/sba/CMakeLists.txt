cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")

#add_definitions(-DSBA_CHOLMOD)
include_directories(../../sba/stereo/extern/ros_sba/include/)

link_directories(/home/kivan/Projects/vista-stereo/build/lib/ros_sba/debug/)
link_directories(/home/kivan/Projects/vista-stereo/build/lib/openMVG/release/lib/)

include_directories(/usr/include/eigen3/)

add_subdirectory(../../sba libs/sba)

file(GLOB SRC_LIST *.cc *.c)
#set(SRC_LIST sba_test.cc)
message(STATUS "${SRC_LIST}")

add_executable(sba_test ${SRC_LIST})
target_link_libraries(sba_test optim_sba)

