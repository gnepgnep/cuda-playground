#!/bin/bash

# Directory path to check
directory="./build"

# Check if the directory exists
if [ -d "$directory" ]; then
    echo "Directory exists"
    rm -rf $directory
else
    echo "Directory does not exist"
fi


# Create build directory if it doesn't exist
mkdir -p build

# Navigate to build directory
cd build

# Generate build files using CMake
cmake -DCMAKE_BUILD_TYPE=debug ..
# cmake .. 

# Compile the project
make
