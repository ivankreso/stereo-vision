#!/bin/bash

#cd /home/kivan/Source/ceres-solver-1.9.0/
cd /home/kivan/Source/ceres-solver-1.9.0/examples/
find . -name "*.cc" -print | xargs sed -i 's/google::ParseCommandLineFlags/gflags::ParseCommandLineFlags/g'
find . -name "*.cc" -print | xargs sed -i 's/google::SetUsageMessage/gflags::SetUsageMessage/g'
