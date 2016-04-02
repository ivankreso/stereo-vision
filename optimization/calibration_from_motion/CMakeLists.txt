cmake_minimum_required(VERSION 2.8)

file(GLOB SRC_LIST . *.cc)

if(NOT TARGET core)
  add_subdirectory(../../core libs/core)
endif()

add_library(optim_cfm ${SRC_LIST})
target_link_libraries(optim_cfm core ceres cxsparse cholmod colamd amd camd
                      suitesparseconfig openblas gfortran gomp glog pthread rt)

