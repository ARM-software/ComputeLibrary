/*
 * Copyright (c) 2021-2023 Arm Limited.
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
#include "src/cpu/operators/CpuGemmConv2d.h"

#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/utils/Log.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/core/helpers/Utils.h"
#include "src/cpu/kernels/CpuCol2ImKernel.h"
#include "src/cpu/kernels/CpuIm2ColKernel.h"
#include "src/cpu/kernels/CpuWeightsReshapeKernel.h"
#include "src/cpu/operators/CpuGemm.h"
#include "src/cpu/operators/CpuGemmLowpMatrixMultiplyCore.h"
#include "src/cpu/operators/CpuGemmLowpOutputStage.h"
#include "src/cpu/operators/CpuReshape.h"
#include "src/cpu/utils/CpuAuxTensorHandler.h"

#include <set>
#include <tuple>

using namespace arm_compute::misc::shape_calculator;
using namespace arm_compute::experimental;

namespace arm_compute
{
namespace cpu
{

/** @section note_CpuGemmConv2d_weight_transformation Weight Transformations in CpuGemmConv2d
 *
 * A. Terminology
 *      Throughout CpuGemmConv2d, we use the following terms in ways that may differ from other operators / kernels:
 *          - "Transform" or "Reshape" of the weights: they both mean all the operations that we perform on the weight
 *             tensor up until they are consumed by gemm (CpuGemm or CpuGemmLowpMatrixMultiplyCore)
 *             Note that the specific gemm operator may perform further transformations on the weights, but the
 *             transformations here only mean those performed in CpuGemmConv2d
 *          - "Transpose" of weights: The @ref CpuTranspose operation. I.e. transpose of the weights' lowest two
 *             dimensions
 *
 * B. Gemm-based conv2d
 *      We want to convert the 2d convolution op (ignoring bias):
 *          dst = conv2d(src, weight)
 *      into a matrix multiplication op:
 *          gemm_dst = gemm(lhs, rhs)
 *
 *      E.g.: For data layout NHWC
 *                               3 (hi) <----------> (lo) 0
 *               src.shape =    [batch,  in_h , in_w,  in_c]
 *               weight.shape = [out_c,   k_h ,  k_w,  in_c]
 *               dst.shape =    [batch, out_h, out_w, out_c]
 *
 *      This requires three transformations:
 *          * src -> lhs, transform conv input to gemm lhs; gemm_lhs is a 2d matrix where each row (or column,
 *                          depending on the convention) is a linearized "patch" of the conv_input that corresponds to
 *                          the receptive field of the corresponding output element.
 *                          The convention is to use "column", but to disambiguate from the column vector of a matrix,
 *                          in this documentation we shall use "patch".
 *                          This transform is called im2col (for details see @ref CpuIm2ColKernel)
 *          * weight -> rhs, transform conv weight to gemm rhs, known as weight transform/reshape (wt)
 *          * gemm_dst -> dst, transform gemm output back to conv output, known as col2im (for details see
 *                          @ref CpuCol2ImKernel)
 *
 *      This section focuses on the weight transformation and assumes the im2col is already performed
 *
 * C. Weight Transformation
 *      After im2col, assume: lhs.shape = [num_patch, patch_size],
 *          where patch_size is the number of elements in a "patch": patch_size = k_h * k_w * in_c
 *                num_patch is the number of patches; we can ignore it here (for details see @ref CpuIm2ColKernel)
 *
 *      After wt, rhs should have the shape: rhs = [patch_size, out_c]
 *
 *      Therefore, the weight transformation consists of two steps:
 *          1. Collapsing all 3 spatial dimensions: [out_c, k_h, k_w, in_c] -> [out_c, patch_size]
 *          2. Transpose the collapsed shape: [out_c, patch_size] -> [patch_size, out_c]
 *
 * D. Implementation
 *      There are 4 paths for weight transformation
 *
 *      1. Path 1: Fixed weight format - no transformation
 *          The underlying gemm kernel may adopt fixed weight format (isVarWeightsKernel() == true), which requires
 *          that no weight transformation shall be performed
 *          Note that this no-transform requirement applies both to this op (CpuGemmConv2d) and the constituent ops, up
 *          until the fixed format kernels themselves
 *
 *      2. Path 2: Reinterpret then transpose later
 *          If the weight tensor has no "holes" (see @ref has_holes), there are two optimizations we can apply:
 *              - We can ignore the first step (collapsing of spatial dimensions) by simply re-interpreting the shape
 *                in TensorInfo
 *              - Instead of performing transpose here, we can pass the transpose flag to the underlying gemm. The gemm
 *                may then decide to fuse the transpose with any further transformations
 *
 *      3. Path 3: Reshape then transpose later
 *          If the weight tensor has holes, then we use a dedicated @ref CpuReshape, followed by transpose later
 *
 *      4. Path 4: Fused reshape and transpose
 *          This is only for quantized types for now (TODO: Remove (COMPMID-6596)). We fall back to a legacy
 *          non-optimized kernel @ref CpuWeightsReshapeKernel to perform a fused reshape + transpose
 *
 *      Path 1 is the long term solution that we shall migrate to once (if) we adopt fixed weight format for all gemm
 *      kernels.
 *      In the short term, Path 2 is the favored, more performant path.
 */

namespace
{
/** Initialize reshaped / transformed weight info
 *
 * @param[in]  weights          Input weights
 * @param[out] reshaped_weights Transformed weights
 */
void initialize_reshaped_weight_info(const ITensorInfo &weights, ITensorInfo &reshaped_weights)
{
    auto_init_if_empty(reshaped_weights, weights);
    if (is_data_type_quantized(weights.data_type()))
    {
        // WT method: FusedReshapeAndTranspose
        reshaped_weights.set_tensor_shape(compute_weights_reshaped_shape(weights, /* has_bias */ false));
    }
    else
    {
        TensorShape collapsed_weights = weights.tensor_shape();
        collapsed_weights.collapse(3);
        reshaped_weights.set_tensor_shape(collapsed_weights);
    }
}
} // namespace

CpuGemmConv2d::WeightTransformMethod CpuGemmConv2d::get_wt_method(const ITensorInfo &weights)
{
    // TODO: Extend ReinterpretThenTranspose support for quantized data types COMPMID-6596
    if (is_data_type_quantized(weights.data_type()))
    {
        return WeightTransformMethod::FusedReshapeAndTranspose;
    }
    return has_holes(weights) ? WeightTransformMethod::ReshapeThenTranspose
                              : WeightTransformMethod::ReinterpretThenTranspose;
}

CpuGemmConv2d::SkipInfo CpuGemmConv2d::skip_im_col_info(const ITensorInfo         *src,
                                                        const ITensorInfo         *weights,
                                                        const PadStrideInfo       &conv_info,
                                                        const Size2D              &dilation,
                                                        const ActivationLayerInfo &act_info)
{
    const DataLayout   data_layout   = src->data_layout();
    const int          idx_width     = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int          idx_height    = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int kernel_width  = weights->dimension(idx_width);
    const unsigned int kernel_height = weights->dimension(idx_height);
    unsigned int       conv_w        = 0;
    unsigned int       conv_h        = 0;
    std::tie(conv_w, conv_h) = scaled_dimensions(src->dimension(idx_width), src->dimension(idx_height), kernel_width,
                                                 kernel_height, conv_info, dilation);
    const bool skip_im2col   = (data_layout == DataLayout::NHWC && kernel_width == 1 && kernel_height == 1 &&
                              conv_info.stride().first == 1 && conv_info.stride().second == 1);

    if (skip_im2col)
    {
        const bool skip_col2im =
            (data_layout == DataLayout::NHWC &&
             (bool(CpuGemmConv2d::validate_gemm3d(src, weights, act_info, conv_h, /*skip_im2col*/ true))));
        if (skip_col2im)
        {
            return {true, true};
        }
    }
    else
    {
        const bool skip_col2im =
            (data_layout == DataLayout::NHWC &&
             (bool(CpuGemmConv2d::validate_gemm3d(src, weights, act_info, conv_h, /*skip_im2col*/ false))));
        if (skip_col2im)
        {
            return {false, true};
        }
    }

    // Default case when we cannot reinterpret the input and output as 3D.
    return {false, false};
}

CpuGemmConv2d::CpuGemmConv2d()
    : _weights_reshape(nullptr),
      _weights_reshape_and_transpose_kernel(nullptr),
      _im2col_kernel(),
      _mm_gemm(),
      _mm_gemmlowp(),
      _col2im_kernel(),
      _reshape(),
      _im2col_output(),
      _weights_reshaped(),
      _gemm_output(),
      _gemm_output_3d(),
      _data_layout(DataLayout::NCHW),
      _skip_im2col(false),
      _skip_col2im(false),
      _is_quantized(false),
      _is_prepared(false),
      _wt_method(WeightTransformMethod::ReshapeThenTranspose),
      _run_wt(true),
      _aux_mem(AuxTensorIdx::Count)
{
}
CpuGemmConv2d::~CpuGemmConv2d() = default;

void CpuGemmConv2d::configure_mm(const ITensorInfo         *src,
                                 const ITensorInfo         *weights,
                                 const ITensorInfo         *biases,
                                 ITensorInfo               *dst,
                                 const ActivationLayerInfo &act_info,
                                 bool                       enable_fast_math,
                                 int                        gemm_3d_depth,
                                 bool                       fixed_format,
                                 arm_compute::WeightFormat  weight_format)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights);
    ARM_COMPUTE_ERROR_THROW_ON(validate_mm(src, weights, biases, dst, act_info, enable_fast_math, gemm_3d_depth,
                                           _skip_im2col, fixed_format, weight_format));

    // Supported activations in GEMM
    const std::set<ActivationLayerInfo::ActivationFunction> supported_acts = {
        ActivationLayerInfo::ActivationFunction::RELU, ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
        ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU};

    if (_is_quantized)
    {
        TensorInfo tmp_src{*src};
        TensorInfo tmp_weights{*weights};
        // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
        // Extract and negate input and weights offset
        const QuantizationInfo        iqinfo    = src->quantization_info();
        const QuantizationInfo        wqinfo    = weights->quantization_info();
        const QuantizationInfo        oqinfo    = (dst->total_size() == 0) ? iqinfo : dst->quantization_info();
        const UniformQuantizationInfo uiqinfo   = iqinfo.uniform();
        const UniformQuantizationInfo uoqinfo   = oqinfo.uniform();
        const DataType                data_type = src->data_type();

        tmp_src.set_quantization_info(QuantizationInfo(uiqinfo.scale, -uiqinfo.offset));
        if (!is_data_type_quantized_per_channel(tmp_weights.data_type()))
        {
            const UniformQuantizationInfo uwqinfo = wqinfo.uniform();
            tmp_weights.set_quantization_info(QuantizationInfo(uwqinfo.scale, -uwqinfo.offset));
        }

        // Merge activation with output stage
        PixelValue type_min{};
        PixelValue type_max{};
        std::tie(type_min, type_max) = get_min_max(data_type);
        int32_t min_activation       = type_min.get<int32_t>();
        int32_t max_activation       = type_max.get<int32_t>();

        if (supported_acts.count(act_info.activation()) != 0)
        {
            std::tie(min_activation, max_activation) = get_quantized_activation_min_max(act_info, data_type, uoqinfo);
        }

        GEMMLowpOutputStageInfo output_info;
        output_info.type                     = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
        output_info.gemmlowp_offset          = uoqinfo.offset;
        output_info.gemmlowp_min_bound       = min_activation;
        output_info.gemmlowp_max_bound       = max_activation;
        output_info.is_quantized_per_channel = (tmp_weights.data_type() == DataType::QSYMM8_PER_CHANNEL);
        quantization::calculate_quantized_multipliers(iqinfo, wqinfo, oqinfo, output_info);

        _mm_gemmlowp = std::make_unique<CpuGemmLowpMatrixMultiplyCore>();
        _mm_gemmlowp->configure(&tmp_src, &tmp_weights, biases, dst,
                                GEMMInfo(false, false, true, gemm_3d_depth, _skip_im2col, false, output_info, false,
                                         enable_fast_math, false, act_info, fixed_format, weight_format,
                                         false /* pretranspose_B. TODO: COMPMID-6596 */));

        auto mm_mem_req = _mm_gemmlowp->workspace();
        for (unsigned int cont = 0; cont < mm_mem_req.size(); ++cont)
        {
            _aux_mem[cont] = mm_mem_req[cont];
        }
    }
    else
    {
        // Create GEMMInfo structure
        const GEMMInfo &gemm_info =
            GEMMInfo(false, false, true /* Reshape weights only for the first run */, gemm_3d_depth,
                     _skip_im2col /* Reinterpret the input as 3D if im2col is skipped */, false,
                     GEMMLowpOutputStageInfo(), false, enable_fast_math, false, act_info, fixed_format, weight_format,
                     true /*pretranspose_B. For fp gemm (wt path 1 - 3), We always pretranspose B (for wt path 1 this
                     flag is ignored)*/);
        // Configure matrix multiply function
        _mm_gemm = std::make_unique<CpuGemm>();
        _mm_gemm->configure(src, weights, biases, dst, 1.0f, 1.0f, gemm_info);
        auto mm_mem_req = _mm_gemm->workspace();
        for (unsigned int cont = 0; cont < mm_mem_req.size(); ++cont)
        {
            _aux_mem[cont] = mm_mem_req[cont];
        }
    }
}

Status CpuGemmConv2d::validate_mm(const ITensorInfo         *src,
                                  const ITensorInfo         *weights,
                                  const ITensorInfo         *biases,
                                  const ITensorInfo         *dst,
                                  const ActivationLayerInfo &act_info,
                                  bool                       enable_fast_math,
                                  int                        gemm_3d_depth,
                                  bool                       skip_im2col,
                                  bool                       fixed_format,
                                  arm_compute::WeightFormat  weight_format)
{
    const DataType data_type             = src->data_type();
    const bool     is_quantized          = is_data_type_quantized_asymmetric(data_type);
    const bool     is_activation_enabled = act_info.enabled();

    if (is_quantized)
    {
        // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
        // Extract and negate input and weights offset
        const QuantizationInfo       &iqinfo  = src->quantization_info();
        const QuantizationInfo       &wqinfo  = weights->quantization_info();
        const QuantizationInfo       &oqinfo  = (dst->total_size() == 0) ? iqinfo : dst->quantization_info();
        const UniformQuantizationInfo uoqinfo = oqinfo.uniform();

        // Merge activation with output stage
        PixelValue type_min{};
        PixelValue type_max{};
        std::tie(type_min, type_max) = get_min_max(data_type);
        int32_t min_activation       = type_min.get<int32_t>();
        int32_t max_activation       = type_max.get<int32_t>();

        const std::set<ActivationLayerInfo::ActivationFunction> supported_acts = {
            ActivationLayerInfo::ActivationFunction::RELU, ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
            ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU};
        if (is_activation_enabled && supported_acts.count(act_info.activation()) != 0)
        {
            std::tie(min_activation, max_activation) = get_quantized_activation_min_max(act_info, data_type, uoqinfo);
        }

        GEMMLowpOutputStageInfo output_info;
        output_info.type                     = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
        output_info.gemmlowp_offset          = uoqinfo.offset;
        output_info.gemmlowp_min_bound       = min_activation;
        output_info.gemmlowp_max_bound       = max_activation;
        output_info.is_quantized_per_channel = (weights->data_type() == DataType::QSYMM8_PER_CHANNEL);
        ARM_COMPUTE_RETURN_ON_ERROR(quantization::calculate_quantized_multipliers(iqinfo, wqinfo, oqinfo, output_info));

        // Perform validation step on GEMMLowp
        std::unique_ptr<ITensorInfo> input_qa   = src->clone();
        std::unique_ptr<ITensorInfo> weights_qa = weights->clone();
        input_qa->set_quantization_info(QuantizationInfo(iqinfo.uniform().scale, -iqinfo.uniform().offset));
        weights_qa->set_quantization_info(QuantizationInfo(wqinfo.uniform().scale, -wqinfo.uniform().offset));

        return CpuGemmLowpMatrixMultiplyCore::validate(input_qa.get(), weights_qa.get(), biases, dst,
                                                       GEMMInfo(false, false, true, gemm_3d_depth, skip_im2col, false,
                                                                output_info, false, enable_fast_math, false, act_info,
                                                                false /* pretranspose_B. TODO: COMPMID-6596 */));
    }
    else
    {
        // Create GEMMInfo structure
        const GEMMInfo gemm_info =
            GEMMInfo(false, false, true /* Reshape weights only for the first run */, gemm_3d_depth,
                     skip_im2col /* Reinterpret the input as 3D if im2col is skipped */, false,
                     GEMMLowpOutputStageInfo(), false, enable_fast_math, false, act_info, fixed_format, weight_format,
                     true /*pretranspose_B. For fp gemm (wt path 1 - 3), We always pretranspose B (for wt path 1 this
                     flag is ignored)*/);

        // Perform validation step on Matrix multiply function
        return CpuGemm::validate(src, weights, biases, dst, 1.0f, 1.0f, gemm_info);
    }
}

Status CpuGemmConv2d::validate_gemm3d(const ITensorInfo         *input_info,
                                      const ITensorInfo         *weights_info,
                                      const ActivationLayerInfo &act_info,
                                      int                        gemm_3d_depth,
                                      bool                       skip_im2col)
{
    const DataType     data_type = input_info->data_type();
    const unsigned int mult_y    = skip_im2col ? 1U : gemm_3d_depth;
    const unsigned int mult_z    = skip_im2col ? gemm_3d_depth : 1U;

    // Set dummy tensor shapes for the validation
    const TensorInfo dummy_input_info(TensorShape(4U, 4U * mult_y, 1U * mult_z), 1, data_type,
                                      input_info->quantization_info());
    const TensorInfo dummy_weights_info(TensorShape(4U, 4U), 1, data_type, weights_info->quantization_info());
    const TensorInfo dummy_output_info(TensorShape(4U, 4U, gemm_3d_depth), 1, data_type,
                                       input_info->quantization_info());

    return validate_mm(&dummy_input_info, &dummy_weights_info, nullptr, &dummy_output_info, act_info, false,
                       gemm_3d_depth, skip_im2col);
}

void CpuGemmConv2d::configure(const ITensorInfo         *src,
                              const ITensorInfo         *weights,
                              const ITensorInfo         *biases,
                              ITensorInfo               *dst,
                              const PadStrideInfo       &conv_info,
                              const WeightsInfo         &weights_info,
                              const Size2D              &dilation,
                              const ActivationLayerInfo &act_info,
                              bool                       enable_fast_math,
                              unsigned int               num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_UNUSED(num_groups, weights_info);
    ARM_COMPUTE_ERROR_THROW_ON(CpuGemmConv2d::validate(src, weights, biases, dst, conv_info, weights_info, dilation,
                                                       act_info, enable_fast_math, num_groups));
    ARM_COMPUTE_LOG_PARAMS(src, weights, biases, dst, conv_info, weights_info, dilation, act_info, enable_fast_math,
                           num_groups);

    const DataType   data_type   = src->data_type();
    const DataLayout data_layout = src->data_layout();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        idx_channel = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);
    const int        idx_kernels = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);

    const unsigned int kernel_width  = weights->dimension(idx_width);
    const unsigned int kernel_height = weights->dimension(idx_height);

    _is_prepared  = weights_info.retain_internal_weights();
    _is_quantized = is_data_type_quantized_asymmetric(src->data_type());
    _data_layout  = data_layout;
    _skip_im2col  = (data_layout == DataLayout::NHWC && kernel_width == 1 && kernel_height == 1 &&
                    conv_info.stride().first == 1 && conv_info.stride().second == 1);

    const ITensorInfo *gemm_input_to_use  = src;
    ITensorInfo       *gemm_output_to_use = dst;

    // Get convolved dimensions
    unsigned int conv_w      = 0;
    unsigned int conv_h      = 0;
    std::tie(conv_w, conv_h) = scaled_dimensions(src->dimension(idx_width), src->dimension(idx_height), kernel_width,
                                                 kernel_height, conv_info, dilation);

    ARM_COMPUTE_ERROR_ON_MSG((dst->dimension(idx_width) != conv_w) || (dst->dimension(idx_height) != conv_h),
                             "Output shape does not match the expected one");

    // Check if GEMM3D is supported
    const CpuGemmConv2d::SkipInfo skip_info =
        CpuGemmConv2d::skip_im_col_info(src, weights, conv_info, dilation, act_info);
    _skip_im2col = skip_info.skip_im2col;
    _skip_col2im = skip_info.skip_col2im;

    // Get parameters from conv_info
    unsigned int stride_x        = 0;
    unsigned int stride_y        = 0;
    std::tie(stride_x, stride_y) = conv_info.stride();

    // Initialize reshaped weights
    initialize_reshaped_weight_info(*weights, _weights_reshaped);

    // Create tensor to store im2col reshaped inputs
    if (!_skip_im2col)
    {
        const int    block_by        = arm_compute::block_by(weights_info.weight_format());
        unsigned int input_pad_right = 0;
        if (block_by > 1)
        {
            input_pad_right =
                (src->dimension(idx_channel) % block_by) == 0 ? 0 : block_by - (src->dimension(idx_channel) % block_by);
        }
        // Configure
        _im2col_kernel = std::make_unique<kernels::CpuIm2ColKernel>();
        _im2col_kernel->configure(src, &_im2col_output, Size2D(kernel_width, kernel_height), conv_info, false, dilation,
                                  num_groups, input_pad_right);

        // Update GEMM input
        gemm_input_to_use = &_im2col_output;
    }

    const unsigned int mat_weights_cols = weights->dimension(idx_kernels);

    // Create temporary GEMM output tensor in case we cannot skip col2im
    const DataType output_data_type = data_type == DataType::BFLOAT16 ? DataType::F32 : data_type;
    if (!_skip_col2im)
    {
        TensorShape shape_gemm;

        // Calculate GEMM output shape
        shape_gemm = _im2col_output.tensor_shape();
        shape_gemm.set(0, mat_weights_cols);
        shape_gemm.set(1, conv_w * conv_h);

        _gemm_output = TensorInfo(shape_gemm, 1, output_data_type);
        _gemm_output.set_quantization_info(dst->quantization_info()).set_data_layout(src->data_layout());
        _gemm_output_3d = TensorInfo(_gemm_output);

        // Update GEMM output
        gemm_output_to_use = &_gemm_output;
    }
    else
    {
        _gemm_output_3d = TensorInfo(*dst);
        _gemm_output_3d.set_data_type(output_data_type).set_data_layout(src->data_layout()).set_is_resizable(true);
        _gemm_output = TensorInfo(_gemm_output_3d);

        // Update GEMM output
        gemm_output_to_use = &_gemm_output_3d;
    }

    // Configure GEMM
    // In case we need to skip col2im, GEMM3D (gemm_3d_depth != 0) must be called in order to avoid reshaping the output matrix
    const unsigned int gemm_3d_depth = _skip_col2im ? conv_h : 0;
    const bool         fixed_format  = weights_info.weight_format() != arm_compute::WeightFormat::UNSPECIFIED;
    /** @section note_CpuGemmConv2d_weight_use_in_configure  Which weights tensor should we use to configure gemm
     *
     *  A. The problem:
     *      In principle, we should use the weights tensor corresponding to the weights transformation path. I.e.:
     *          - If no weight transformation (_run_wt == false): Use original weights
     *          - else:                                           Use transformed weights
     *      However in practice we have a dilemma:
     *          - We need to know _run_wt before we can configure gemm with the corresponding weights, but
     *          - _run_wt depends on isVarWeightsKernel(), which is only known after gemm is configured
     *
     *  B. The decision:
     *      To simplify the matter, we decide to always use the transformed weights, regardless of _run_wt
     *
     *      This decision requires the following conditions:
     *          1. The underlying gemm where isVarWeightsKernel() == true, must guarantee that:
     *              A. Ignore the flag to transpose weights (GEMMInfo::pretranspose_B)
     *              B. Use weights/B tensor passed to it at prepare() or run() instead of that passed at configure()
     *          2. CpuGemmConv2d where isVarWeightsKernel() == true, must guarantee that:
     *              A. Pass original weights instead of reshaped or reinterpreted weights
     *
     *  C. Future actions:
     *      Condition 2 is a given, based on our implementation.
     *      If condition 1 cannot hold, we must make changes to the underlying gemm to:
     *           1. Either expose isVarWeightsKernel() before gemm is configured somehow, or
     *           2. Take in an additional "original_weights" tensor info at configure
     */
    configure_mm(gemm_input_to_use, &_weights_reshaped, biases, gemm_output_to_use, act_info, enable_fast_math,
                 gemm_3d_depth, fixed_format, weights_info.weight_format());

    // Can only decide isVarWeightsKernel after gemm is configured
    _run_wt = !isVarWeightsKernel();

    if (!_skip_col2im && _data_layout == DataLayout::NCHW)
    {
        // Configure col2im
        _col2im_kernel = std::make_unique<kernels::CpuCol2ImKernel>();
        _col2im_kernel->configure(gemm_output_to_use, dst, Size2D(conv_w, conv_h));
    }
    else
    {
        // Configure reshape layer
        _reshape = std::make_unique<CpuReshape>();
        _reshape->configure(gemm_output_to_use, dst);
    }

    // Check lifetime
    _aux_mem[Im2ColOutput] =
        MemoryInfo(offset_int_vec(Im2ColOutput), MemoryLifetime::Temporary, _im2col_output.total_size());
    // Add WeightsReshaped memory requirement to workspace
    // Note that in case of WeightTransformMethod::ReinterpretThenTranspose, we do not need to allocate this memory
    // However since we cannot determine weight transformation method until prepare (see prepare()), we will have to
    // settle with allocating more
    if (_run_wt)
    {
        // Check if GEMM transforms weights
        // If weight is further transformed by underlying gemm after ReshapeThenTranspose then we can free
        // WeightsReshaped in prepare
        // Otherwise WeightsReshaped is the final transformation of weights and needs to persist
        bool gemm_trans_wei = _aux_mem[GemmAsmPretransposedRHS].size > 0;
        gemm_trans_wei      = _mm_gemm != nullptr ? _aux_mem[GemmTransposed1xWRHS].size > 0 : gemm_trans_wei;
        gemm_trans_wei      = _mm_gemmlowp != nullptr ? _aux_mem[GemmLowpTransposed1xWRHS].size > 0 : gemm_trans_wei;

        _aux_mem[WeightsReshaped] = MemoryInfo(offset_int_vec(WeightsReshaped),
                                               gemm_trans_wei ? MemoryLifetime::Prepare : MemoryLifetime::Persistent,
                                               _weights_reshaped.total_size());
    }
    _aux_mem[GemmOutput] = MemoryInfo(offset_int_vec(GemmOutput), MemoryLifetime::Temporary, _gemm_output.total_size());
}

Status CpuGemmConv2d::has_opt_impl(arm_compute::WeightFormat &expected_weight_format,
                                   const ITensorInfo         *src,
                                   const ITensorInfo         *weights,
                                   const ITensorInfo         *biases,
                                   const ITensorInfo         *dst,
                                   const PadStrideInfo       &conv_info,
                                   const WeightsInfo         &weights_info,
                                   const Size2D              &dilation,
                                   const ActivationLayerInfo &act_info,
                                   const bool                 enable_fast_math)
{
    const DataLayout   data_layout   = src->data_layout();
    const int          idx_width     = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int          idx_height    = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int kernel_width  = weights->dimension(idx_width);
    const unsigned int kernel_height = weights->dimension(idx_height);
    unsigned int       conv_w        = 0;
    unsigned int       conv_h        = 0;
    std::tie(conv_w, conv_h) = scaled_dimensions(src->dimension(idx_width), src->dimension(idx_height), kernel_width,
                                                 kernel_height, conv_info, dilation);

    const CpuGemmConv2d::SkipInfo skip_info =
        CpuGemmConv2d::skip_im_col_info(src, weights, conv_info, dilation, act_info);

    const bool         skip_im2col   = skip_info.skip_im2col;
    const bool         skip_col2im   = skip_info.skip_col2im;
    const unsigned int gemm_3d_depth = skip_col2im ? conv_h : 0;
    const bool         fixed_format  = weights_info.weight_format() != arm_compute::WeightFormat::UNSPECIFIED;

    /** @section note_CpuGemmConv2d_weight_use_in_has_opt_impl Which weights tensor should we use for has_opt_impl
     *
     *  For the pretranspose_B flag, this shares a similar problem and thus the same decision as that of
     *  @ref note_CpuGemmConv2d_weight_use_in_configure
     *
     *  But for the weights, we shall always use the original instead of reshaped weights here
     */
    const GEMMInfo gemm_info = GEMMInfo(false, false, true /* Reshape weights only for the first run */, gemm_3d_depth,
                                        skip_im2col /* Reinterpret the input as 3D if im2col is skipped */, false,
                                        GEMMLowpOutputStageInfo(), false, enable_fast_math, false, act_info,
                                        fixed_format, weights_info.weight_format(), true /* pretranspose_B */);

    return CpuGemm::has_opt_impl(expected_weight_format, src, weights, biases, dst, gemm_info);
}

Status CpuGemmConv2d::validate(const ITensorInfo         *src,
                               const ITensorInfo         *weights,
                               const ITensorInfo         *biases,
                               const ITensorInfo         *dst,
                               const PadStrideInfo       &conv_info,
                               const WeightsInfo         &weights_info,
                               const Size2D              &dilation,
                               const ActivationLayerInfo &act_info,
                               bool                       enable_fast_math,
                               unsigned int               num_groups)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights_info.are_reshaped(), "Weights already reshaped are not supported!");
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::BFLOAT16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::QSYMM8_PER_CHANNEL, DataType::BFLOAT16,
                                                         DataType::F16, DataType::F32);

    if (!is_fixed_format(weights_info.weight_format()))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(src, weights);
    }

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(num_groups > 1, "Grouping (num_groups != 1) is not supported");

    const DataLayout data_layout = src->data_layout();
    const DataType   data_type   = src->data_type();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        idx_channel = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);
    const int        idx_kernels = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);

    const unsigned int kernel_width  = weights->dimension(idx_width);
    const unsigned int kernel_height = weights->dimension(idx_height);

    TensorInfo         im2col_reshaped_info{};
    TensorInfo         info_gemm{};
    TensorInfo         tmp_info{};
    TensorInfo         weights_reshaped_info{};
    const ITensorInfo *gemm_input_to_use  = src;
    const ITensorInfo *gemm_output_to_use = dst;
    const ITensorInfo *weights_to_use     = weights;

    const bool append_bias  = false;
    const bool is_quantized = is_data_type_quantized_asymmetric(data_type);
    const bool is_bf16      = data_type == DataType::BFLOAT16;

    // Get convolved dimensions
    unsigned int conv_w = 0;
    unsigned int conv_h = 0;

    std::tie(conv_w, conv_h) = scaled_dimensions(src->dimension(idx_width), src->dimension(idx_height), kernel_width,
                                                 kernel_height, conv_info, dilation);

    // Check if GEMM3D is supported
    const CpuGemmConv2d::SkipInfo skip_info =
        CpuGemmConv2d::skip_im_col_info(src, weights, conv_info, dilation, act_info);
    const bool skip_im2col = skip_info.skip_im2col, skip_col2im = skip_info.skip_col2im;

    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_channel) != src->dimension(idx_channel));
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);

    // Validate biases
    if (biases != nullptr)
    {
        if (is_quantized)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::S32);
        }
        else if (is_bf16)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::F32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, biases);
        }
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != dst->dimension(idx_channel));
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    }

    unsigned int mat_weights_cols = weights->dimension(idx_kernels);
    unsigned int mat_weights_rows =
        weights->dimension(idx_width) * weights->dimension(idx_height) * weights->dimension(idx_channel);

    // Initialize reshaped weights
    initialize_reshaped_weight_info(*weights, weights_reshaped_info);
    // No need to call CpuReshape::validate() or CpuTranspose::validate() as the dst info is auto-configured from the
    // src
    weights_to_use = &weights_reshaped_info;

    if (!skip_im2col)
    {
        const int block_by        = arm_compute::block_by(weights_info.weight_format());
        int       input_pad_right = 0;
        if (block_by > 1)
        {
            input_pad_right =
                (src->dimension(idx_channel) % block_by) == 0 ? 0 : block_by - (src->dimension(idx_channel) % block_by);
            mat_weights_rows = weights->dimension(idx_width) * weights->dimension(idx_height) *
                               (weights->dimension(idx_channel) + input_pad_right);
        }

        // Create tensor info for im2col reshaped inputs
        // For CPU, the batch size is on the fourth dimension
        TensorShape shape_im2col = src->tensor_shape();
        shape_im2col.set(0, mat_weights_rows);
        shape_im2col.set(1, conv_w * conv_h);
        shape_im2col.set(2, 1);

        im2col_reshaped_info = TensorInfo(shape_im2col, 1, data_type);
        im2col_reshaped_info.set_quantization_info(src->quantization_info());
        ARM_COMPUTE_RETURN_ON_ERROR(
            kernels::CpuIm2ColKernel::validate(src, &im2col_reshaped_info, Size2D(kernel_width, kernel_height),
                                               conv_info, append_bias, dilation, num_groups, input_pad_right));
        gemm_input_to_use = &im2col_reshaped_info;
    }

    // Create temporary GEMM output tensor in case we cannot skip col2im
    const DataType output_data_type = data_type == DataType::BFLOAT16 ? DataType::F32 : data_type;
    if (!skip_col2im)
    {
        TensorShape shape_gemm = gemm_input_to_use->tensor_shape();
        shape_gemm.set(0, mat_weights_cols);
        shape_gemm.set(1, conv_w * conv_h);
        info_gemm = TensorInfo(shape_gemm, 1, output_data_type);
    }
    else
    {
        info_gemm = TensorInfo(dst->tensor_shape(), 1, output_data_type);
    }
    info_gemm.set_quantization_info(dst->quantization_info()).set_data_layout(src->data_layout());
    gemm_output_to_use      = &info_gemm;
    const bool fixed_format = weights_info.weight_format() != arm_compute::WeightFormat::UNSPECIFIED;

    // See note_CpuGemmConv2d_weight_use_in_configure regarding the choice of the weights
    ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(gemm_input_to_use, weights_to_use, biases, gemm_output_to_use, act_info,
                                            enable_fast_math, skip_col2im ? conv_h : 0, skip_im2col, fixed_format,
                                            weights_info.weight_format()));

    // Validate Col2Im/ReshapeLayer
    if (!skip_col2im && (data_layout == DataLayout::NCHW))
    {
        ARM_COMPUTE_RETURN_ON_ERROR(
            kernels::CpuCol2ImKernel::validate(gemm_output_to_use, dst, Size2D(conv_w, conv_h)));
    }

    return Status{};
}

void CpuGemmConv2d::run(ITensorPack &tensors)
{
    prepare(tensors);

    auto src               = tensors.get_const_tensor(ACL_SRC_0);
    auto dst               = tensors.get_tensor(ACL_DST);
    auto gemm_input_to_use = src;

    CpuAuxTensorHandler im2col_output(offset_int_vec(Im2ColOutput), _im2col_output, tensors, false);
    CpuAuxTensorHandler gemm_output(offset_int_vec(GemmOutput), _gemm_output, tensors, false);

    bool out_has_padding = _skip_col2im && (dst->info()->padding().bottom != 0 || dst->info()->padding().top != 0);
    if (!_skip_im2col)
    {
        // Run input reshaping
        unsigned int y_dim = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);
        ITensorPack  pack  = {{TensorType::ACL_SRC, src}, {TensorType::ACL_DST, im2col_output.get()}};
        NEScheduler::get().schedule_op(_im2col_kernel.get(), y_dim, _im2col_kernel->window(), pack);
        gemm_input_to_use = im2col_output.get();
    }

    // Handle the case where output has top/bottom padding
    const ITensor *out_to_use = out_has_padding ? gemm_output.get() : dst;
    Tensor         gemm3d;
    _gemm_output_3d.extend_padding(out_to_use->info()->padding());
    gemm3d.allocator()->soft_init(_gemm_output_3d);
    gemm3d.allocator()->import_memory(out_to_use->buffer());
    auto gemm_output_to_use = gemm_output.get();

    if (_skip_im2col)
    {
        gemm_output_to_use = &gemm3d;
    }
    if (_skip_col2im && !out_has_padding)
    {
        gemm_output_to_use = dst;
    }

    ITensorPack gemm_pack = tensors;
    gemm_pack.add_const_tensor(TensorType::ACL_SRC_0, gemm_input_to_use);
    gemm_pack.add_tensor(TensorType::ACL_DST, gemm_output_to_use);
    // Allocate reshaped weights if required
    auto weights = gemm_pack.get_const_tensor(TensorType::ACL_SRC_1);
    ARM_COMPUTE_ERROR_ON_NULLPTR(weights);
    // Re-interpreted weights. Only tensor shape is changed. Only memory import, no allocation
    CpuAuxTensorHandler reinterpreted_wei(
        _weights_reshaped, *weights,
        /* import only if we chose the ReinterpretThenTranspose path, because otherwise the weight may have been freed */
        !(_run_wt && _wt_method == WeightTransformMethod::ReinterpretThenTranspose));
    CpuAuxTensorHandler reshaped_wei(offset_int_vec(WeightsReshaped), _weights_reshaped, tensors);
    // Update the weights to use if it has been reshaped
    if (_run_wt)
    {
        if (_wt_method == WeightTransformMethod::ReinterpretThenTranspose)
        {
            gemm_pack.add_const_tensor(TensorType::ACL_SRC_1, reinterpreted_wei.get());
        }
        else if (_wt_method == WeightTransformMethod::ReshapeThenTranspose ||
                 _wt_method == WeightTransformMethod::FusedReshapeAndTranspose)
        {
            gemm_pack.add_const_tensor(TensorType::ACL_SRC_1, reshaped_wei.get());
        }
    }

    // Runs CpuGemm or CpuGemmLowpMatrixMultiplyCore functions
    _is_quantized ? _mm_gemmlowp->run(gemm_pack) : _mm_gemm->run(gemm_pack);

    // Reshape output matrix
    if (!_skip_col2im)
    {
        if (_data_layout == DataLayout::NCHW)
        {
            ITensorPack pack = {{TensorType::ACL_SRC, gemm_output.get()}, {TensorType::ACL_DST, dst}};
            NEScheduler::get().schedule_op(_col2im_kernel.get(), Window::DimY, _col2im_kernel->window(), pack);
        }
        else
        {
            ITensorPack pack = {{TensorType::ACL_SRC, gemm_output_to_use}, {TensorType::ACL_DST, dst}};
            _reshape->run(pack);
        }
    }
    else if (out_has_padding)
    {
        ITensorPack pack = {{TensorType::ACL_SRC, gemm_output_to_use}, {TensorType::ACL_DST, dst}};
        _reshape->run(pack);
    }
}

void CpuGemmConv2d::prepare(ITensorPack &tensors)
{
    if (!_is_prepared)
    {
        auto weights = tensors.get_const_tensor(TensorType::ACL_SRC_1);
        // Determine which weights reshape path to take
        // Note that this decision can only occur at prepare instead of configure because it relies on the presence of
        // any holes in the weight tensor, which may change after configure (e.g. from extending padding)
        if (_run_wt)
        {
            _wt_method = get_wt_method(*(weights->info()));
            switch (_wt_method)
            {
                case (WeightTransformMethod::FusedReshapeAndTranspose):
                {
                    ARM_COMPUTE_LOG_INFO_WITH_FUNCNAME_ACL("Perform weight transformation: FusedReshapeAndTranspose");
                    _weights_reshape_and_transpose_kernel = std::make_unique<kernels::CpuWeightsReshapeKernel>();
                    _weights_reshape_and_transpose_kernel->configure(weights->info(), nullptr, &_weights_reshaped);
                    break;
                }
                case (WeightTransformMethod::ReshapeThenTranspose):
                {
                    ARM_COMPUTE_LOG_INFO_WITH_FUNCNAME_ACL("Perform weight transformation: ReshapeThenTranspose");
                    _weights_reshape = std::make_unique<CpuReshape>();
                    _weights_reshape->configure(weights->info(), &_weights_reshaped);
                    break;
                }
                case (WeightTransformMethod::ReinterpretThenTranspose):
                {
                    ARM_COMPUTE_LOG_INFO_WITH_FUNCNAME_ACL("Perform weight transformation: ReinterpretThenTranspose");
                    // Nothing to configure
                    break;
                }
                default:
                {
                    ARM_COMPUTE_ERROR("Unsupported weight transform method");
                }
            }
        }
        else
        {
            ARM_COMPUTE_LOG_INFO_WITH_FUNCNAME_ACL("No weight transformation is performed");
        }
        ITensorPack gemm_pack = tensors;
        // Allocate reshaped weights if required
        CpuAuxTensorHandler reinterpreted_wei(
            _weights_reshaped,
            *weights); // Re-interpreted weights. Only tensor shape is changed. No allocation
        CpuAuxTensorHandler reshaped_wei(offset_int_vec(WeightsReshaped), _weights_reshaped, tensors);
        // Run weights reshape if required
        if (_run_wt)
        {
            switch (_wt_method)
            {
                case (WeightTransformMethod::FusedReshapeAndTranspose):
                {
                    ITensorPack pack = {{TensorType::ACL_SRC, weights}, {TensorType::ACL_DST, reshaped_wei.get()}};
                    NEScheduler::get().schedule_op(_weights_reshape_and_transpose_kernel.get(), Window::DimW,
                                                   _weights_reshape_and_transpose_kernel->window(), pack);
                    weights->mark_as_unused();
                    gemm_pack.add_const_tensor(TensorType::ACL_SRC_1, reshaped_wei.get());
                    break;
                }
                case (WeightTransformMethod::ReshapeThenTranspose):
                {
                    ITensorPack pack = {{TensorType::ACL_SRC, weights}, {TensorType::ACL_DST, reshaped_wei.get()}};
                    _weights_reshape->run(pack);
                    weights->mark_as_unused();
                    gemm_pack.add_const_tensor(TensorType::ACL_SRC_1, reshaped_wei.get());
                    break;
                }
                case (WeightTransformMethod::ReinterpretThenTranspose):
                {
                    gemm_pack.add_const_tensor(TensorType::ACL_SRC_1, reinterpreted_wei.get());
                    // Nothing to run
                    break;
                }
                default:
                {
                    ARM_COMPUTE_ERROR("Unsupported weight transform method");
                }
            }
        }
        _is_quantized ? _mm_gemmlowp->prepare(gemm_pack) : _mm_gemm->prepare(gemm_pack);

        _is_prepared = true;
    }
}
experimental::MemoryRequirements CpuGemmConv2d::workspace() const
{
    return _aux_mem;
}
bool CpuGemmConv2d::isVarWeightsKernel() const
{
    return _mm_gemm && _mm_gemm->isVarWeightsKernel();
}
} // namespace cpu
} // namespace arm_compute
