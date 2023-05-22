# Compute Kernel Writer

Project description to follow.

## Getting started


### Building and running tests

The fastest way to get started with Compute Kernel Writer is to build and run the test suite.

#### Compile natively on Linux x86_64

```shell
mkdir build && cd build
CC=gcc CXX=g++ cmake -G Ninja -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCKW_ENABLE_OPENCL=ON -DCKW_ENABLE_ASSERTS=ON -DCKW_BUILD_TESTING=ON ..
cmake --build .
```

#### Cross-compile to Linux aarch64

```shell
mkdir build && cd build
cmake -G Ninja -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCKW_ENABLE_OPENCL=ON -DCKW_ENABLE_ASSERTS=ON -DCKW_BUILD_TESTING=ON -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/gcc_linux_aarch64.toolchain.cmake ..
cmake --build .
```

#### Cross-compile to Android aarch64

Cross-compiling to the Android platform requires the toolchain CMake file downloaded in the [Android NDK](https://developer.android.com/ndk).

```shell
mkdir build && cd build
cmake -G Ninja -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCKW_ENABLE_OPENCL=ON -DCKW_ENABLE_ASSERTS=ON -DCKW_BUILD_TESTING=ON -DCMAKE_TOOLCHAIN_FILE=<NDK>/build/cmake/android.toolchain.cmake ..
cmake --build .
```

#### Run the validation suite

```shell
./ckw_validation
```
