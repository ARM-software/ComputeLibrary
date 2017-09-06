#!/usr/bin/env bash

DIRECTORIES="./arm_compute ./src ./examples ./tests ./utils ./support"

if [ $# -eq 0 ]
then
    files=$(find $DIRECTORIES -type f -name \*.cpp | sort)
else
    files=$@
fi

SCRIPT_PATH=$(dirname $0)

CLANG_TIDY=$(which clang-tidy)

if [[ -z $CLANG_TIDY ]]; then
    echo "clang-tidy not found!"
    exit 1
else
    echo "Found clang-tidy:" $CLANG_TIDY
fi

CLANG_TIDY_PATH=$(dirname $CLANG_TIDY)/..

ARMV7_GCC=$(which arm-linux-gnueabihf-g++)

if [[ -z $ARMV7_GCC ]]; then
    echo "arm-linux-gnueabihf-g++ not found!"
    exit 1
else
    echo "Found arm-linux-gnueabihf-g++:" $ARMV7_GCC
fi

ARMV7_GCC_PATH=$(dirname $ARMV7_GCC)/..

AARCH64_GCC=$(which aarch64-linux-gnu-g++)

if [[ -z $AARCH64_GCC ]]; then
    echo "aarch64-linux-gnu-g++ not found!"
    exit 1
else
    echo "Found aarch64-linux-gnu-g++:" $AARCH64_GCC
fi

ARMV7_GCC_PATH=$(dirname $ARMV7_GCC)/..
AARCH64_GCC_PATH=$(dirname $AARCH64_GCC)/..

INCLUDE_PATHS="-Iinclude -I. -I3rdparty/include -Ikernels -Icomputer_vision"

function armv7
{
    USE_BOOST=""

    if [[ "$1" == *tests/validation_old* ]]
    then
        USE_BOOST="-DBOOST"
    fi

    $CLANG_TIDY \
    "$1" \
    -- \
    -target armv7a-none-linux-gnueabihf \
    --gcc-toolchain=$ARMV7_GCC_PATH \
    -std=c++11 \
    $INCLUDE_PATHS \
    -DARM_COMPUTE_CPP_SCHEDULER=1 $USE_BOOST
    #read -rsp $'Press enter to continue...\n'
}

function aarch64
{
    USE_BOOST=""

    if [[ "$1" == *tests/validation_old* ]]
    then
        USE_BOOST="-DBOOST"
    fi

    $CLANG_TIDY \
    "$1" \
    -- \
    -target aarch64-none-linux-gnueabi \
    --gcc-toolchain=$AARCH64_GCC_PATH \
    -std=c++11 \
    -include $SCRIPT_PATH/clang-tidy.h \
    $INCLUDE_PATHS \
    -DARM_COMPUTE_CL -DARM_COMPUTE_ENABLE_FP16 -DARM_COMPUTE_CPP_SCHEDULER=1 $USE_BOOST
}

for f in $files; do
    #armv7 "$f"
    aarch64 "$f"
done
