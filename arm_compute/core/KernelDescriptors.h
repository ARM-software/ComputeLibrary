/*
 * Copyright (c) 2019 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef __ARM_COMPUTE_CORE_KERNEL_DESCRIPTORS_H__
#define __ARM_COMPUTE_CORE_KERNEL_DESCRIPTORS_H__

#include "arm_compute/core/Types.h"

namespace arm_compute
{
/** Descriptor for FFT scale kernels */
struct FFTScaleKernelInfo
{
    float scale{ 0.f };      /**< Axis to perform the kernel on. */
    bool  conjugate{ true }; /**< Flag to conjugate the output/ */
};

/** Descriptor for FFT digit reverse kernels */
struct FFTDigitReverseKernelInfo
{
    unsigned int axis{ 0 };          /**< Axis to perform the kernel on. */
    bool         conjugate{ false }; /**< Flag to conjugate the output/ */
};

/** Descriptor used by the FFT core kernels */
struct FFTRadixStageKernelInfo
{
    unsigned int axis{ 0 };               /**< Axis to run the kernel on. */
    unsigned int radix{ 0 };              /**< Radix to use. */
    unsigned int Nx{ 0 };                 /**< Nx coefficient. */
    bool         is_first_stage{ false }; /**< Flags if the FFT kernels is the first stage of a decomposed FFT. */
};

/** Descriptor used by the GEMM kernels */
struct GEMMKernelInfo
{
    unsigned int        m{ 0 };                           /**< Number of LHS rows*/
    unsigned int        n{ 0 };                           /**< Number of RHS columns*/
    unsigned int        k{ 0 };                           /**< Number of LHS columns or RHS rows */
    unsigned int        depth_output_gemm3d{ 0 };         /**< Depth of the output tensor in case is reinterpreted as 3D */
    bool                reinterpret_input_as_3d{ false }; /**< Flag used to reinterpret the input as 3D */
    bool                broadcast_bias{ false };          /**< Flag used to broadcase the bias addition */
    ActivationLayerInfo activation_info{};                /**< Activation function to perform after the matrix multiplication */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CORE_KERNEL_DESCRIPTORS_H__ */
