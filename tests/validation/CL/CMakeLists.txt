# Copyright (c) 2017 ARM Limited.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
cmake_minimum_required (VERSION 3.1)

include_directories(${CMAKE_SOURCE_DIR}/../include)

set(arm_compute_test_validation_OPENCL_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/CL/CLAccessor.h
    ${CMAKE_SOURCE_DIR}/CL/Helper.h
    ${CMAKE_CURRENT_SOURCE_DIR}/CLFixture.h
    ${CMAKE_CURRENT_SOURCE_DIR}/CLFixture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BitwiseAnd.cpp
`   ${CMAKE_CURRENT_SOURCE_DIR}/IntegralImage.cpp
)

add_library(arm_compute_test_validation_OPENCL OBJECT
    ${arm_compute_test_validation_OPENCL_SOURCE_FILES}
)

set(arm_compute_test_validation_TARGET_OBJECTS
    ${arm_compute_test_validation_TARGET_OBJECTS}
    $<TARGET_OBJECTS:arm_compute_test_validation_OPENCL>
    PARENT_SCOPE
)

set(arm_compute_test_validation_TARGET_LIBRARIES
    ${arm_compute_test_validation_TARGET_LIBRARIES}
    OpenCL
    PARENT_SCOPE
)
