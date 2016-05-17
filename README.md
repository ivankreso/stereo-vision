# Stereo Structure and Motion (SaM / SfM) library

## Egomotion
* Implemented using Ceres Solver, two basic variants with Quaternions and Euler angles
* Examples in `stereo_egomotion/main`

## Dense stereo
* Implementation of Semi-global matching in `reconstruction/base`
* Examples in `reconstruction/main`

## Feature tracking
* Monocular and steresopic variants.
* Implementation in `tracker` directory

## Publications

* [Improving the Egomotion Estimation by Correcting the Calibration Bias](http://www.cvlibs.net/datasets/kitti/eval_odometry_detail.php?&result=3ef2e95144c13778b66cec9b1d4c887c68684cea)
* Code in `deformation_field`
