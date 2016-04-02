cmake_minimum_required(VERSION 2.8)

#include_directories(/home/kivan/Source/ceres-solver-1.9.0/internal/)
#include_directories(/home/kivan/Source/ceres-solver-1.9.0/include/)
#include_directories(/home/kivan/Source/ceres-solver-1.9.0/build/config/)
include_directories(/home/kivan/Source/ceres-solver/internal/)
include_directories(/home/kivan/Source/ceres-solver/include/)
include_directories(/home/kivan/Source/bin/ceres-solver/config/)

file(GLOB SRC_LIST . *.cc)

add_library(optim_bundle_adjustment ${SRC_LIST})

# use glog
target_link_libraries(optim_bundle_adjustment opencv_core ceres cxsparse cholmod colamd amd camd suitesparseconfig
                      libopenblas.a gfortran gomp glog pthread rt)

# use miniglog
#include_directories(/home/kivan/Source/ceres-solver-1.9.0/internal/ceres/miniglog/)
#target_link_libraries(optim_sba opencv_core ceres cxsparse cholmod colamd amd camd suitesparseconfig
#                      libopenblas.a gfortran gomp miniglog pthread rt)
#target_link_libraries(optim_sba opencv_core ceres cxsparse cholmod colamd amd camd suitesparseconfig
#                      openblas gfortran gomp miniglog pthread rt)

#target_link_libraries(optim_sba opencv_core ceres cxsparse cholmod colamd amd camd suitesparseconfig
#                      lapack blas cblas gomp atlas miniglog pthread rt)
#target_link_libraries(optim_sba opencv_core ceres cxsparse cholmod colamd amd camd suitesparseconfig
#                      lapack cblas f77blas gomp gfortran atlas glog pthread rt)

# use MKL
#target_link_libraries(optim_sba opencv_core ceres cxsparse cholmod colamd amd camd suitesparseconfig
#                      mkl_intel_lp64 mkl_core mkl_intel_thread iomp5 pthread miniglog rt)
