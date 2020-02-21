/*
 * Copyright (c) 2019-2020 ARM Limited.
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
#ifndef ARM_COMPUTE_CORE_KERNEL_DESCRIPTORS_H
#define ARM_COMPUTE_CORE_KERNEL_DESCRIPTORS_H

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
    bool                fp_mixed_precision{ false };      /**< Flag used to indicate wider accumulators (32 bit instead of 16 for FP16). */
    ActivationLayerInfo activation_info{};                /**< Activation function to perform after the matrix multiplication */
};

/** Descriptor used by the depthwise convolution kernels */
struct DWCKernelInfo
{
    ActivationLayerInfo activation_info{}; /**< Activation function to perform after the depthwise convolution */
};

/** Descriptor used by the depthwise convolution kernels to retrieve the number of output elements processed by each thread */
struct DWCWeightsKernelInfo
{
    unsigned int n0{ 0 }; /**< Number of columns processed by each thread */
};

/** Descriptor used by the softmax kernels */
struct SoftmaxKernelInfo
{
    float    beta{ 1.f };                          /**< A scaling factor for the exponent with default value 1.0 */
    bool     is_log{ false };                      /**< Flag used to perform Log Softmax operation */
    DataType input_data_type{ DataType::UNKNOWN }; /**< Input tensor data type */
};

/** Descriptor used by the direct convolution layer output stage kernels */
struct DirectConvolutionLayerOutputStageKernelInfo
{
    int32_t  result_fixedpoint_multiplier{ 0 };     /**< Result output stage multiplier used for quantizing */
    int32_t  result_shift{ 0 };                     /**< Result output stage shift used for quantizing */
    int32_t  result_offset_after_shift{ 0 };        /**< Result offset used for quantizing */
    DataType output_data_type{ DataType::UNKNOWN }; /**< Output tensor data type to use if the output is not initialized */
};

struct InstanceNormalizationLayerKernelInfo
{
    /** Default constructor */
    InstanceNormalizationLayerKernelInfo()
        : InstanceNormalizationLayerKernelInfo(1.f, 0.f, 1e-12, true)
    {
    }
    /** Constructor
     *
     * @param[in] gamma               The scale scalar value applied to the normalized tensor.
     * @param[in] beta                The offset scalar value applied to the normalized tensor
     * @param[in] epsilon             Lower bound value for the normalization.
     * @param[in] use_mixed_precision Use mixed precision in case of FP16 execution.
     */
    InstanceNormalizationLayerKernelInfo(float gamma, float beta, float epsilon, bool use_mixed_precision)
        : gamma(gamma), beta(beta), epsilon(epsilon), use_mixed_precision(use_mixed_precision)
    {
    }

    float gamma;               /**< The scale scalar value applied to the normalized tensor. Defaults to 1.0 */
    float beta;                /**< The offset scalar value applied to the normalized tensor. Defaults to 0.0 */
    float epsilon;             /**< Lower bound value for the normalization. Defaults to 1e-12 */
    bool  use_mixed_precision; /**< Use mixed precision in case of FP16 execution. Defaults to true */
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CORE_KERNEL_DESCRIPTORS_H */
