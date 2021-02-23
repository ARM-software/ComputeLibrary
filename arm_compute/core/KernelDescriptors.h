/*
 * Copyright (c) 2019-2021 Arm Limited.
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

#include "arm_compute/core/PixelValue.h"
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
    GEMMKernelInfo() = default;
    GEMMKernelInfo(
        unsigned int        im,
        unsigned int        in,
        unsigned int        ik,
        unsigned int        idepth_output_gemm3d,
        bool                ireinterpret_input_as_3d,
        bool                ibroadcast_bias,
        bool                ifp_mixed_precision,
        bool                ihas_pad_y,
        ActivationLayerInfo iactivation_info,
        int                 inmult_transpose1xW_width,
        int                 imult_interleave4x4_height,
        GEMMLHSMatrixInfo   ilhs_info,
        GEMMRHSMatrixInfo   irhs_info,
        int32_t             ina_offset,
        int32_t             inb_offset)
        : m(im), n(in), k(ik), depth_output_gemm3d(idepth_output_gemm3d), reinterpret_input_as_3d(ireinterpret_input_as_3d), broadcast_bias(ibroadcast_bias), fp_mixed_precision(ifp_mixed_precision),
          has_pad_y(ihas_pad_y), activation_info(iactivation_info), mult_transpose1xW_width(inmult_transpose1xW_width), mult_interleave4x4_height(imult_interleave4x4_height), lhs_info(ilhs_info),
          rhs_info(irhs_info), a_offset(ina_offset), b_offset(inb_offset)
    {
    }

    unsigned int            m{ 0 };                           /**< Number of LHS rows*/
    unsigned int            n{ 0 };                           /**< Number of RHS columns*/
    unsigned int            k{ 0 };                           /**< Number of LHS columns or RHS rows */
    unsigned int            depth_output_gemm3d{ 0 };         /**< Depth of the output tensor in case is reinterpreted as 3D */
    bool                    reinterpret_input_as_3d{ false }; /**< Flag used to reinterpret the input as 3D */
    bool                    broadcast_bias{ false };          /**< Flag used to broadcast the bias addition */
    bool                    fp_mixed_precision{ false };      /**< Flag used to indicate wider accumulators (32 bit instead of 16 for FP16). */
    bool                    has_pad_y{ false };               /**< Flag used to indicate if the input/output tensors have internal pad on the y direction */
    ActivationLayerInfo     activation_info{};                /**< Activation function to perform after the matrix multiplication */
    int                     mult_transpose1xW_width{ 1 };     /**< Multiplication factor for the width of the 1xW transposed block */
    int                     mult_interleave4x4_height{ 1 };   /**< Multiplication factor for the height of the 4x4 interleaved block */
    GEMMLHSMatrixInfo       lhs_info{};                       /**< LHS matrix information used to retrieve the number of rows processed by each thread */
    GEMMRHSMatrixInfo       rhs_info{};                       /**< RHS matrix information used for reshaping the RHS matrix */
    int32_t                 a_offset{ 0 };                    /**< Offset to be added to each element of the matrix A */
    int32_t                 b_offset{ 0 };                    /**< Offset to be added to each element of the matrix B */
    GEMMLowpOutputStageInfo output_stage{};                   /**< GEMMLowp output stage information */
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

struct GEMMLowpReductionKernelInfo
{
    /** Default constructor */
    GEMMLowpReductionKernelInfo() = default;
    /** Constructor
     *
     * @param[in] k             Number of matrix columns/rows.
     * @param[in] is_reshaped   True if the input tensor has been reshaped.
     * @param[in] scalar        Scalar value to multiply each reduced column/row by.
     * @param[in] mul_by_scalar True if each column/row reduction has to be multiplied by a scalar value.
     */
    GEMMLowpReductionKernelInfo(int32_t k, bool is_reshaped, int32_t scalar, bool mul_by_scalar)
        : k(k), is_reshaped(is_reshaped), scalar(scalar), mul_by_scalar(mul_by_scalar)
    {
    }

    int32_t k{ 0 };                 /**< Number of matrix columns/rows */
    bool    is_reshaped{ false };   /**< True if the input tensor has been reshaped */
    int32_t scalar{ 0 };            /**< Scalar value to multiply each reduced column/row by */
    bool    mul_by_scalar{ false }; /**< True if each column/row reduction has to be multiplied by a scalar value */
};

struct ScaleKernelInfo
{
    /** Constructor
     *
     * @param[in] interpolation_policy  Interpolation type to use
     * @param[in] border_mode           Border mode policy
     * @param[in] constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT and use_padding is set to false. Defaults to default @ref PixelValue
     * @param[in] sampling_policy       (Optional) Sampling policy used by the interpolation. Defaults to @ref SamplingPolicy::CENTER
     * @param[in] use_padding           (Optional) Is padding in use or not. Defaults to true.
     * @param[in] align_corners         (Optional) Align corners of input and output, only affecting bilinear policy with TOP_LEFT sampling policy. Defaults to false.
     * @param[in] data_layout           (Optional) Data layout used by the layer. Defaults to @ref DataLayout::UNKNOWN
     */
    ScaleKernelInfo(InterpolationPolicy interpolation_policy,
                    BorderMode          border_mode,
                    PixelValue          constant_border_value = PixelValue(),
                    SamplingPolicy      sampling_policy       = SamplingPolicy::CENTER,
                    bool                use_padding           = true,
                    bool                align_corners         = false,
                    DataLayout          data_layout           = DataLayout::UNKNOWN)
        : interpolation_policy{ interpolation_policy },
          border_mode{ border_mode },
          constant_border_value{ constant_border_value },
          sampling_policy{ sampling_policy },
          use_padding{ use_padding },
          align_corners{ align_corners },
          data_layout{ data_layout }
    {
    }

    InterpolationPolicy interpolation_policy;  /**< Interpolation type to use */
    BorderMode          border_mode;           /**< Border mode policy */
    PixelValue          constant_border_value; /**< Constant value to use for constant border mode policy */
    SamplingPolicy      sampling_policy;       /**< Sampling policy used by the interpolation. */
    bool                use_padding;           /**< Indication of using padding */
    bool                align_corners;         /**< Align corners of input and output */
    DataLayout          data_layout;           /**< Data layout to use */
};

struct ThresholdKernelInfo
{
    /** Default constructor */
    ThresholdKernelInfo() = default;
    /** Constructor
     *
     * @param[in] threshold   Threshold. When the threshold type is RANGE, this is used as the lower threshold.
     * @param[in] false_value value to set when the condition is not respected.
     * @param[in] true_value  value to set when the condition is respected.
     * @param[in] type        Thresholding type. Either RANGE or BINARY.
     * @param[in] upper       Upper threshold. Only used when the thresholding type is RANGE.
     */
    ThresholdKernelInfo(uint8_t threshold, uint8_t false_value, uint8_t true_value, ThresholdType type, uint8_t upper)
        : threshold(threshold), false_value(false_value), true_value(true_value), type(type), upper(upper)
    {
    }

    uint8_t       threshold{ 0 };
    uint8_t       false_value{ 0 };
    uint8_t       true_value{ 0 };
    ThresholdType type{ ThresholdType::BINARY };
    uint8_t       upper{ 0 };
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CORE_KERNEL_DESCRIPTORS_H */
