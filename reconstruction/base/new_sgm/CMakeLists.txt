cmake_minimum_required(VERSION 2.8)
project(SGM)

set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")
#set(CMAKE_CXX_FLAGS "-std=c++11 -fsanitize=memory -fno-omit-frame-pointer -fno-optimize-sibling-calls -g -O1")
#set(CMAKE_CXX_FLAGS "-std=c++11 -fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls -g -O1")

include_directories(/usr/include/eigen3/)
include_directories(/home/kivan/libraries/install/include/)
link_directories(/home/kivan/libraries/install/lib/)

# Flags
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-O1 -g")

file(GLOB SRC_FILES "*.cc")
add_executable(sgm ${SRC_FILES})
#target_link_libraries(sgm png opencv_core opencv_imgproc opencv_highgui)
target_link_libraries(sgm png opencv_core opencv_imgcodecs opencv_imgproc)

