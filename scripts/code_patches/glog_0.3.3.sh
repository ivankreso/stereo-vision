#!/bin/bash

cd /home/kivan/Source/glog-0.3.3/src/
find . -name "*.cc" -print | xargs sed -i 's/ParseCommandLineFlags/gflags::ParseCommandLineFlags/g'
find . -name "*.cc" -print | xargs sed -i 's/FlagSaver/gflags::FlagSaver/g'
find . -name "*.h" -print | xargs sed -i 's/FlagSaver/gflags::FlagSaver/g'
