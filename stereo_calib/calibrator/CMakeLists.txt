# run example: (in build dir)
# in Debug/ folder:
# cmake ../../StereoVision/ -DCMAKE_BUILD_TYPE=Debug
# in Release/ folder:
# cmake ../../StereoVision/ -DCMAKE_BUILD_TYPE=Release
cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")

include_directories(/home/kivan/libraries/install/include/)
link_directories(/home/kivan/libraries/install/lib)

set(SRC_CALIBRATOR_STEREO_INTR calibrator_stereo_intr.cc)
set(SRC_CALIBRATOR_STEREO calibrator_stereo.cc)
set(SRC_CALIBRATOR_MONO calibrator_mono.cc)
set(SRC_CALIBRATOR_MONO_OLD calibrator_mono_old.cc)
set(SRC_IMAGELIST_GEN imagelist_creator.cc)
set(OPENCV_LIBS opencv_core opencv_imgproc opencv_calib3d opencv_highgui opencv_features2d opencv_videoio opencv_imgcodecs)

#add_executable(calibrator_stereo_intr ${SRC_CALIBRATOR_STEREO_INTR})
<<<<<<< HEAD
=======

>>>>>>> ab2cdf689aa408bb6d2c1384409fe31a106527bc
add_executable(calibrator_stereo ${SRC_CALIBRATOR_STEREO})
add_executable(calibrator_mono ${SRC_CALIBRATOR_MONO})
add_executable(calibrator_mono_old ${SRC_CALIBRATOR_MONO_OLD})
add_executable(imagelist_creator ${SRC_IMAGELIST_GEN})

#target_link_libraries(calibrator_stereo_intr ${OPENCV_LIBS})
target_link_libraries(calibrator_stereo ${OPENCV_LIBS})
target_link_libraries(calibrator_mono ${OPENCV_LIBS})
target_link_libraries(calibrator_mono_old ${OPENCV_LIBS})
target_link_libraries(imagelist_creator opencv_core)

