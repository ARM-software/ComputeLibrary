# Compute Kernel Writer

Compute Kernel Writer is a tile-based, just-in-time code writer for deep learning and computer vision applications.
This tool offers a C++ interface to allow developers to write functions without a return type (called "kernels")
using their preferred programming language (at the moment, only OpenCL is supported). 
The library is specifically designed to be lightweight and to offer an intuitive API for efficient code writing.

## Getting started

The fastest way to get started with Compute Kernel Writer is to build and run the test suite.
The following subsections show you how to do this.

### Dependencies

This project requires the following dependencies, obtainable via your preferred package manager, to be installed and available on your system.

* `build-essential`
* `cmake >= 3.14`
* (Optional) `ninja-build`

In addition, the guide makes use of the following toolchains:

* (Optional) `Arm GNU toolchain` available to download from the [Arm Developer](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads) website
* (Optional) `Android NDK toolset` available to download from the [Android Developer](https://developer.android.com/ndk/downloads/index.html) website

### Building and running tests

#### Native compilation

You can quickly compile the library on your computer by using the following commands:

```shell
mkdir -p build && cd build
CXX=g++ cmake -G Ninja -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCKW_ENABLE_OPENCL=ON -DCKW_ENABLE_ASSERTS=ON -DCKW_BUILD_TESTING=ON -S ..
cmake --build .
```

The preceding commands build the library in release mode (`-DCMAKE_BUILD_TYPE=Release`) and targets OpenCL code generation (`-DCKW_ENABLE_OPENCL=ON`). 
In addition, code assertions are enabled (`-DCKW_ENABLE_ASSERTS=ON`) and the test suite is built (`-DCKW_BUILD_TESTING=ON`). 
Alternatively, choose to build a static instead of a shared library by setting `-DBUILD_SHARED_LIBS=OFF`.

#### Cross-compile to Linux AArch64

The Arm GNU toolchain can be used to cross-compile the project to a Linux system with an AArch64 processor, like a Raspberry Pi, using an x86_64 Linux host machine.

```shell
mkdir -p build && cd build
CXX=aarch64-none-linux-gnu-g++ cmake -G Ninja -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCKW_ENABLE_OPENCL=ON -DCKW_ENABLE_ASSERTS=ON -DCKW_BUILD_TESTING=ON -S ..
cmake --build .
```

The build configuration is identical to the previous step but now requires specifying the target triple in the CXX compiler (`CXX=aarch64-none-linux-gnu-g++`) to generate binaries for the target platform.

#### Cross-compile to Android AArch64

Cross-compiling for Android systems requires the Android NDK toolset. The downloaded NDK contains the toolchain file necessary for cross-compiling the project.

```shell
mkdir -p build && cd build
cmake -G Ninja -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCKW_ENABLE_OPENCL=ON -DCKW_ENABLE_ASSERTS=ON -DCKW_BUILD_TESTING=ON -DCMAKE_TOOLCHAIN_FILE=<NDK>/build/cmake/android.toolchain.cmake -S ..
cmake --build .
```

This build re-uses the same build configuration as before, but this time does not require specifying the CXX compiler as this (and other target-specific information) is handled by the toolchain file (`-DCMAKE_TOOLCHAIN_FILE`).

#### Run the validation test suite

Confirm the project has been built successfully by running the validation test suite.

```shell
./ckw_validation
```

### List of build options

This project can be configured with the following build options. Enable options by passing them to the CMake command, preceded with `-D`.

| Option               | Description                                                                                                                               |
|:---------------------|:------------------------------------------------------------------------------------------------------------------------------------------|
| BUILD_SHARED_LIBS    | Controls whether to build static or shared libraries.                                                                                     |
| CMAKE_BUILD_TYPE     | The project build type or configuration. Choose from Release or Debug. <br/>The release build will always build for smallest binary size. |
| CKW_ENABLE_OPENCL    | Enable OpenCL code generation.                                                                                                            |
| CKW_ENABLE_ASSERTS   | Enable assertions. Always enabled for Debug builds.                                                                                       |
| CKW_BUILD_TESTING    | Build the validation test suite.                                                                                                          |
| CKW_CCACHE           | Use compiler cache for faster recompilation.                                                                                              |
| CMAKE_TOOLCHAIN_FILE | When cross-compiling, set this variable to the path of the CMake toolchain file.                                                          |
