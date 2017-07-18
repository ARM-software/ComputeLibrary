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

set(arm_compute_test_validation_NEON_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/NEON/Helper.h
    ${CMAKE_SOURCE_DIR}/NEON/Accessor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/AbsoluteDifference.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Accumulate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AccumulateSquared.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AccumulateWeighted.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ArithmeticAddition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ArithmeticSubtraction.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BatchNormalizationLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BitwiseAnd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BitwiseNot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BitwiseOr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BitwiseXor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Box3x3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ConvolutionLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ConvolutionLayerDirect.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DepthConvert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FillBorder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Fixedpoint/Exp_QS8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Fixedpoint/Invsqrt_QS8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Fixedpoint/Log_QS8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Fixedpoint/Reciprocal_QS8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FullyConnectedLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Gaussian3x3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GEMM.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/IntegralImage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NormalizationLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PixelWiseMultiplication.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MeanStdDev.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Pooling/PoolingLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Sobel3x3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Sobel5x5.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SoftmaxLayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Threshold.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MinMaxLocation.cpp
)

add_library(arm_compute_test_validation_NEON OBJECT
    ${arm_compute_test_validation_NEON_SOURCE_FILES}
)

set(arm_compute_test_validation_TARGET_OBJECTS
    ${arm_compute_test_validation_TARGET_OBJECTS}
    $<TARGET_OBJECTS:arm_compute_test_validation_NEON>
    PARENT_SCOPE
)
