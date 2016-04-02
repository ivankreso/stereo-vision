cmake_minimum_required(VERSION 2.8)
find_package(OpenCV COMPONENTS opencv_core opencv_highgui opencv_calib3d REQUIRED)

# setup boost library
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED OFF)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.53.0 COMPONENTS program_options thread system)
message(STATUS "${Boost_FOUND}")
if(NOT ${Boost_FOUND})
   message(FATAL_ERROR "boost lib not found!")
endif()

file(GLOB SRC_LIST *.cc)
set(SRC_LIST ${SRC_LIST})

message(STATUS "${OpenCV_LIBS}")
message(STATUS "${Boost_LIBRARIES}")

include_directories(/usr/include/eigen3/)

add_subdirectory(../../../core/ libs/core/)
add_subdirectory(../../base/ libs/base/)

add_executable(dense_stereo ${SRC_LIST})
target_link_libraries(dense_stereo ${Boost_LIBRARIES} ${OpenCV_LIBS} core recon_base vwCore vwImage vwMath vwFileIO)
