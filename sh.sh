#!/bin/bash

BIN_NAME="nn"
TEST_NAME="tests"
BUILD_DIR="build"
DATA_DIR="data"
PREPARED_DATA_DIR="$DATA_DIR/prepared"
WEIGHTS_DIR="$DATA_DIR/weights"

cmd_build() {
    if [ -d "$DATA_DIR" ]; then 
        echo "$DATA_DIR/ dir detected"
    else 
        echo "creating $DATA_DIR/ dir..."
        mkdir -p "$DATA_DIR"
    fi
    if [ -d "$BUILD_DIR" ]; then 
        echo "$BUILD_DIR/ dir detected"
    else 
        echo "creating $BUILD_DIR/ dir..."
        mkdir -p "$BUILD_DIR"
    fi
    cd "$BUILD_DIR"
    echo "configuring..."
    cmake -DCMAKE_CXX_COMPILER=g++ ..
    echo "building..."
    make -j$(nproc)
    cd ..
    echo "Build complete. Python module is in $BUILD_DIR/"
}

cmd_clean() {
    echo "cleaning..."
    rm -rf "$BUILD_DIR"
    rm -rf "$PREPARED_DATA_DIR"
    rm -rf "$WEIGHTS_DIR"
    echo "done"
}

case "$1" in
    build)
        cmd_build
        ;;
    clean)
        cmd_clean
        ;;
    *)
        echo "Usage: $0 {build|clean}"
        exit 1
        ;;
esac