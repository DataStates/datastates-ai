#!/bin/bash

CMAKE_PREFIX="$HOME/deploy/share/cmake;$HOME/deploy/lib/cmake"

cmake -DCMAKE_INSTALL_PREFIX=$HOME/deploy -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX\
      -DCMAKE_BUILD_TYPE=Debug -G "Unix Makefiles" ..

