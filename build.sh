#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build

# Navigate to build directory
cd build

# Generate build files using CMake
cmake ..

# Compile the project
make
