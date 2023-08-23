/*
 * Copyright (c) 2016-2023 Arm Limited.
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
#ifndef ACL_ARM_COMPUTE_FUNCTION_INFO_GEMMINFO
#define ACL_ARM_COMPUTE_FUNCTION_INFO_GEMMINFO

#include "arm_compute/core/CoreTypes.h"
#include "arm_compute/core/experimental/IPostOp.h"
#include "arm_compute/function_info/ActivationLayerInfo.h"
#include <vector>

namespace arm_compute
{
class ITensorInfo;
/** GEMMLowp output stage type */
enum class GEMMLowpOutputStageType
{
    NONE,                     /**< No quantization */
    QUANTIZE_DOWN,            /**< Quantize using an integer multiplication */
    QUANTIZE_DOWN_FIXEDPOINT, /**< Quantize using a fixed point multiplication */
    QUANTIZE_DOWN_FLOAT       /**< Quantize using a floating point multiplication */
};

/** GEMMLowp output stage info */
struct GEMMLowpOutputStageInfo
{
    GEMMLowpOutputStageType type{ GEMMLowpOutputStageType::NONE };                        /**< GEMMLowp output stage type */
    int32_t                 gemmlowp_offset{ 0 };                                         /**< GEMMLowp output stage offset used for quantizing to QASYMM8 */
    int32_t                 gemmlowp_multiplier{ 0 };                                     /**< GEMMLowp output stage multiplier used for quantizing to QASYMM8 */
    int32_t                 gemmlowp_shift{ 0 };                                          /**< GEMMLowp output stage shift used for quantizing to uint8 */
    int32_t                 gemmlowp_min_bound{ std::numeric_limits<int32_t>::lowest() }; /**< GEMMLowp min value used to saturate down the output result before converting back to QASYMM8 */
    int32_t                 gemmlowp_max_bound{ std::numeric_limits<int32_t>::max() };    /**< GEMMLowp max value used to saturate down the output result before converting back to QASYMM8 */
    std::vector<int32_t>    gemmlowp_multipliers{};                                       /**< GEMMLowp output stage multiplier used for quantizing to QASYMM8 */
    std::vector<int32_t>    gemmlowp_shifts{};                                            /**< GEMMLowp output stage multiplier used for quantizing to QASYMM8 */
    float                   gemmlowp_real_multiplier{ 0 };                                /**< GEMMLowp output stage real multiplier used for quantizing to QASYMM8 */
    bool                    is_quantized_per_channel{ false };                            /**< GEMMLowp quantized per-channel flag */
    DataType                output_data_type{ DataType::UNKNOWN };                        /**< Output tensor data type to use if the output is not initialized */
};
/** GEMM information class. This class stores the necessary information to compute GEMM functions
 *
 * This object also contains the information about how matrix A and matrix B have been reshaped
 *
 */
class GEMMInfo
{
public:
    /** Default constructor */
    GEMMInfo() noexcept
        : _is_a_reshaped(false),
          _is_b_reshaped(false),
          _reshape_b_only_on_first_run(true),
          _depth_output_gemm3d(0),
          _reinterpret_input_as_3d(false),
          _retain_internal_weights(false),
          _gemmlowp_output_stage(),
          _fast_math(false),
          _fp_mixed_precision(false),
          _broadcast_bias(false),
          _pretranspose_A(false),
          _pretranspose_B(false),
          _activation_info(),
          _post_ops(),
          _fixed_format(false),
          _weight_format(arm_compute::WeightFormat::UNSPECIFIED)
    {
    }
    /** Constructor
     *
     * @param[in] is_a_reshaped               True if the matrix A has been reshaped
     * @param[in] is_b_reshaped               True if the matrix B has been reshaped
     * @param[in] reshape_b_only_on_first_run Reshape matrix B only for the first run
     * @param[in] depth_output_gemm3d         (Optional) Depth (third dimension) of the output tensor to be used with the GEMM3D kernel
     *                                        If 0 the output will not be reinterpreted as 3D. Default 0
     * @param[in] reinterpret_input_as_3d     (Optional) Reinterpret the input as 3D tensor. (i.e. this flag should be set to true when GEMM is used
     *                                        to perform 1x1 convolutions with the NHWC data layout)
     * @param[in] retain_internal_weights     (Optional) Retain the weights tensor from previous run
     * @param[in] gemmlowp_output_stage       (Optional) GEMMLowp Output stage info
     * @param[in] fp_mixed_precision          (Optional) Use wider accumulators (32 bit instead of 16 for FP16) to improve accuracy.
     * @param[in] fast_math                   (Optional) Use a data type of shorter width to improve performance
     * @param[in] broadcast_bias              (Optional) Broadcast the shape of the bias tensor from a vector to a matrix.
     * @param[in] activation_info             (Optional) Activation to apply after the matrix multiplication
     * @param[in] post_ops                    (Optional) A sequence of post operations that are performed after the main operation.
     * @param[in] fixed_format                (Optional) Specify the selection of fixed format kernels for variable weights support in GEMM. These kernels expect the weights tensor to be in amemory format that is fixed by the kernel itself. For more information, see arm_compute::WeightFormat.
     * @param[in] weight_format               (Optional) arm_gemm:WeightFormat enumeration requested by the user. Default is arm_compute::WeightFormat::UNSPECIFIED.
     */
    GEMMInfo(bool is_a_reshaped, bool is_b_reshaped, bool reshape_b_only_on_first_run, int depth_output_gemm3d = 0, bool reinterpret_input_as_3d = false, bool retain_internal_weights = false,
             GEMMLowpOutputStageInfo gemmlowp_output_stage = GEMMLowpOutputStageInfo(), bool fp_mixed_precision = false, bool fast_math = false, bool broadcast_bias = false,
             const ActivationLayerInfo &activation_info = ActivationLayerInfo(), const experimental::PostOpList<ITensorInfo *> &post_ops = experimental::PostOpList<ITensorInfo *>(),
             bool fixed_format = false, arm_compute::WeightFormat weight_format = arm_compute::WeightFormat::UNSPECIFIED) noexcept
        : _is_a_reshaped(is_a_reshaped),
          _is_b_reshaped(is_b_reshaped),
          _reshape_b_only_on_first_run(reshape_b_only_on_first_run),
          _depth_output_gemm3d(depth_output_gemm3d),
          _reinterpret_input_as_3d(reinterpret_input_as_3d),
          _retain_internal_weights(retain_internal_weights),
          _gemmlowp_output_stage(gemmlowp_output_stage),
          _fast_math(fast_math),
          _fp_mixed_precision(fp_mixed_precision),
          _broadcast_bias(broadcast_bias),
          _pretranspose_A(false),
          _pretranspose_B(false),
          _activation_info(activation_info),
          _post_ops(post_ops),
          _fixed_format(fixed_format),
          _weight_format(weight_format)
    {
    }
    /** Flag which specifies if the matrix A has been reshaped
     *
     * @return True if the matrix A has been reshaped
     */
    bool is_a_reshaped() const
    {
        return _is_a_reshaped;
    };
    /** Flag which specifies if the matrix B has been reshaped
     *
     * @return True if the matrix B has been reshaped
     */
    bool is_b_reshaped() const
    {
        return _is_b_reshaped;
    };
    /** Flag which specifies if the reshape of matrix B should executed only for the first
     *
     * @note This flag could be set to TRUE when GEMM is used to accelerate convolution layer
     *
     * @return True if the reshaped of matrix B happens only for the first run
     */
    bool reshape_b_only_on_first_run() const
    {
        return _reshape_b_only_on_first_run;
    };
    /** Depth of the output when GEMM output is reinterpreted as 3D tensor
     *
     * @return the depth of the output tensor
     */
    int depth_output_gemm3d() const
    {
        return _depth_output_gemm3d;
    };
    /** Flag which specifies if the input tensor has to be reinterpreted as 3D
     *
     * @return True if the input tensor has to be reinterpreted as 3D tensor
     */
    bool reinterpret_input_as_3d() const
    {
        return _reinterpret_input_as_3d;
    };
    /** Flag which specifies if the weights tensor has to be retained from previous run
     *
     * @return True if the weights tensor has to be retained
     */
    bool retain_internal_weights() const
    {
        return _retain_internal_weights;
    };
    /** GEMMLowp output stage
     *
     * @return the GEMMLowp output stage info
     */
    GEMMLowpOutputStageInfo gemmlowp_output_stage() const
    {
        return _gemmlowp_output_stage;
    };
    /** Sets GEMMLowp output stage
     *
     * @param[in] output_stage Output stage to set
     */
    void set_gemmlowp_output_stage(GEMMLowpOutputStageInfo &output_stage)
    {
        _gemmlowp_output_stage = output_stage;
    };
    /** Flag which specifies if a wider accumulator should be used.
     *
     * @return True if a wider accumulator has to be used
     */
    bool fp_mixed_precision() const
    {
        return _fp_mixed_precision;
    };
    /** Flag which specifies if a shorter accumulator to be used.
     *
     * @return True if a shorter accumulator has to be used
     */
    bool fast_math() const
    {
        return _fast_math;
    };
    /** Set fast math flag
     *
     * @param[in] fast_math Flag to set
     */
    void set_fast_math(bool fast_math)
    {
        _fast_math = fast_math;
    }
    /** Flag which specifies whether to broadcast the shape of the bias tensor.
     *
     * @return True if the shape of the bias tensor is to be broadcasted.
     */
    bool broadcast_bias() const
    {
        return _broadcast_bias;
    };
    /** Flag which specifies whether A should be pre-transposed if supported.
     *
     * @return True if A should be pre-transposed else false.
     */
    bool pretranspose_A() const
    {
        return _pretranspose_A;
    };
    /** Set pre-transpose A flag
     *
     * @param[in] flag Flag to set
     */
    void set_pretranspose_A(bool flag)
    {
        _pretranspose_A = flag;
    }
    /** Flag which specifies whether b should be pre-transposed if supported.
     *
     * @return True if b should be pre-transposed else false.
     */
    bool pretranspose_B() const
    {
        return _pretranspose_B;
    };
    /** Set pre-transpose b flag
     *
     * @param[in] flag Flag to set
     */
    void set_pretranspose_B(bool flag)
    {
        _pretranspose_B = flag;
    }
    /** Activation layer to apply after the matrix multiplication
     *
     * @return ActivationLayerInfo object
     */
    ActivationLayerInfo activation_info() const
    {
        return _activation_info;
    }
    /** Set activation layer info
     *
     * @param[in] activation_info ActivationLayerInfo object to set
     */
    void set_activation_info(const ActivationLayerInfo &activation_info)
    {
        _activation_info = activation_info;
    }
    /** Post operations to apply after the matrix multiplication
     *
     * @return experimental::PostOpList object
     */
    const experimental::PostOpList<ITensorInfo *> &post_ops() const
    {
        return _post_ops;
    }
    /** Set post ops
     *
     * @param[in] post_ops experimental::PostOpList object to set
     */
    void set_post_ops(const experimental::PostOpList<ITensorInfo *> &post_ops)
    {
        _post_ops = post_ops;
    }
    /** Flag which specifies if the GEMM operation is running fixed-format kernels.
     *
     * @return True if the GEMM operation is running fixed-format kernel else false.
     */
    bool fixed_format() const
    {
        return _fixed_format;
    }

    /** Set fixed-format flag
     *
     * @param[in] fixed_format sets whether or not to use fixed-format kernels
     */
    void set_fixed_format(bool fixed_format)
    {
        _fixed_format = fixed_format;
    }

    arm_compute::WeightFormat weight_format() const
    {
        return _weight_format;
    }

    /** Set weight format to be used
     *
     * @param[in] weight_format arm_compute::WeightFormat enumeration
     */
    void set_weight_format(arm_compute::WeightFormat weight_format)
    {
        _weight_format = weight_format;
    }

private:
    bool                                    _is_a_reshaped;
    bool                                    _is_b_reshaped;
    bool                                    _reshape_b_only_on_first_run;
    int                                     _depth_output_gemm3d;
    bool                                    _reinterpret_input_as_3d;
    bool                                    _retain_internal_weights;
    GEMMLowpOutputStageInfo                 _gemmlowp_output_stage;
    bool                                    _fast_math;
    bool                                    _fp_mixed_precision;
    bool                                    _broadcast_bias;
    bool                                    _pretranspose_A;
    bool                                    _pretranspose_B;
    ActivationLayerInfo                     _activation_info;
    experimental::PostOpList<ITensorInfo *> _post_ops;
    bool                                    _fixed_format;
    arm_compute::WeightFormat               _weight_format;
};
} //namespace arm_compute
#endif /* ACL_ARM_COMPUTE_FUNCTION_INFO_GEMMINFO */
