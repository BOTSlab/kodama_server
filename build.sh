#!/bin/bash -l

rm -fr build
mkdir -p build
cd build
cmake ..
make
cd ..