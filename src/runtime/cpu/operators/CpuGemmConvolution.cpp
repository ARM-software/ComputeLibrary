/*
 * Copyright (c) 2021 Arm Limited.
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
#include "src/runtime/cpu/operators/CpuGemmConvolution.h"

#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/core/cpu/kernels/CpuCol2ImKernel.h"
#include "src/core/cpu/kernels/CpuIm2ColKernel.h"
#include "src/core/cpu/kernels/CpuReshapeKernel.h"
#include "src/core/cpu/kernels/CpuWeightsReshapeKernel.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/runtime/cpu/operators/CpuGemm.h"
#include "src/runtime/cpu/operators/CpuGemmLowpMatrixMultiplyCore.h"
#include "src/runtime/cpu/operators/CpuGemmLowpOutputStage.h"
#include "src/runtime/cpu/utils/CpuAuxTensorHandler.h"

#include <set>
#include <tuple>

using namespace arm_compute::misc::shape_calculator;
using namespace arm_compute::experimental;

namespace arm_compute
{
namespace cpu
{
CpuGemmConvolution::CpuGemmConvolution()
    : _weights_reshape_kernel(nullptr), _im2col_kernel(), _mm_gemm(), _mm_gemmlowp(), _col2im_kernel(), _reshape_kernel(), _im2col_output(), _weights_reshaped(), _gemm_output(), _gemm_output_3d(),
      _data_layout(DataLayout::NCHW), _skip_im2col(false), _skip_col2im(false), _is_quantized(false), _is_prepared(false), _aux_mem(AuxTensorIdx::Count)
{
}
CpuGemmConvolution::~CpuGemmConvolution() = default;

void CpuGemmConvolution::configure_mm(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst, const ActivationLayerInfo &act_info,
                                      bool enable_fast_math, int gemm_3d_depth)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights);
    ARM_COMPUTE_ERROR_THROW_ON(validate_mm(src, weights, biases, dst, act_info, enable_fast_math, gemm_3d_depth, _skip_im2col));

    // Create GEMMInfo structure
    const GEMMInfo &gemm_info = GEMMInfo(false, false, true /* Reshape weights only for the first run */,
                                         gemm_3d_depth, _skip_im2col /* Reinterpret the input as 3D if im2col is skipped */,
                                         false, GEMMLowpOutputStageInfo(), false, enable_fast_math, false, act_info);

    // Supported activations in GEMM
    const std::set<ActivationLayerInfo::ActivationFunction> supported_acts = { ActivationLayerInfo::ActivationFunction::RELU,
                                                                               ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
                                                                               ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU
                                                                             };

    if(_is_quantized)
    {
        TensorInfo tmp_src{ *src };
        TensorInfo tmp_weights{ *weights };
        // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
        // Extract and negate input and weights offset
        const QuantizationInfo        iqinfo    = src->quantization_info();
        const QuantizationInfo        wqinfo    = weights->quantization_info();
        const QuantizationInfo        oqinfo    = (dst->total_size() == 0) ? iqinfo : dst->quantization_info();
        const UniformQuantizationInfo uiqinfo   = iqinfo.uniform();
        const UniformQuantizationInfo uoqinfo   = oqinfo.uniform();
        const DataType                data_type = src->data_type();

        tmp_src.set_quantization_info(QuantizationInfo(uiqinfo.scale, -uiqinfo.offset));
        if(!is_data_type_quantized_per_channel(tmp_weights.data_type()))
        {
            const UniformQuantizationInfo uwqinfo = wqinfo.uniform();
            tmp_weights.set_quantization_info(QuantizationInfo(uwqinfo.scale, -uwqinfo.offset));
        }

        // Merge activation with output stage
        PixelValue type_min{};
        PixelValue type_max{};
        std::tie(type_min, type_max) = get_min_max(data_type);
        int32_t min_activation = type_min.get<int32_t>();
        int32_t max_activation = type_max.get<int32_t>();

        if(supported_acts.count(act_info.activation()) != 0)
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
        _mm_gemmlowp->configure(&tmp_src, &tmp_weights, biases, dst, GEMMInfo(false, false, true, gemm_3d_depth, _skip_im2col, false, output_info, false, enable_fast_math, false, act_info));

        auto mm_mem_req = _mm_gemmlowp->workspace();
        for(unsigned int cont = 0; cont < mm_mem_req.size(); ++cont)
        {
            _aux_mem[cont] = mm_mem_req[cont];
        }
    }
    else
    {
        // Configure matrix multiply function
        _mm_gemm = std::make_unique<CpuGemm>();
        _mm_gemm->configure(src, weights, biases, dst, 1.0f, 0.0f, gemm_info);
        auto mm_mem_req = _mm_gemm->workspace();
        for(unsigned int cont = 0; cont < mm_mem_req.size(); ++cont)
        {
            _aux_mem[cont] = mm_mem_req[cont];
        }
    }
}

Status CpuGemmConvolution::validate_mm(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst,
                                       const ActivationLayerInfo &act_info, bool enable_fast_math, int gemm_3d_depth, bool skip_im2col)
{
    const DataType data_type             = src->data_type();
    const bool     is_quantized          = is_data_type_quantized_asymmetric(data_type);
    const bool     is_activation_enabled = act_info.enabled();

    // Create GEMMInfo structure
    const GEMMInfo gemm_info = GEMMInfo(false, false, true /* Reshape weights only for the first run */,
                                        gemm_3d_depth, skip_im2col /* Reinterpret the input as 3D if im2col is skipped */,
                                        false, GEMMLowpOutputStageInfo(), false, enable_fast_math, false, act_info);

    if(is_quantized)
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
        int32_t min_activation = type_min.get<int32_t>();
        int32_t max_activation = type_max.get<int32_t>();

        const std::set<ActivationLayerInfo::ActivationFunction> supported_acts = { ActivationLayerInfo::ActivationFunction::RELU,
                                                                                   ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
                                                                                   ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU
                                                                                 };
        if(is_activation_enabled && supported_acts.count(act_info.activation()) != 0)
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
        return CpuGemmLowpMatrixMultiplyCore::validate(input_qa.get(), weights_qa.get(), biases, dst, GEMMInfo(false, false, true, gemm_3d_depth, skip_im2col, false, output_info,
                                                                                                               false, enable_fast_math, false, act_info));
    }
    else
    {
        // Perform validation step on Matrix multiply function
        return CpuGemm::validate(src, weights, nullptr, dst, 1.0f, 0.0f, gemm_info);
    }
}

Status CpuGemmConvolution::validate_gemm3d(const ITensorInfo *input_info, const ITensorInfo *weights_info, const ActivationLayerInfo &act_info, int gemm_3d_depth, bool skip_im2col)
{
    const DataType     data_type = input_info->data_type();
    const unsigned int mult_y    = skip_im2col ? 1U : gemm_3d_depth;
    const unsigned int mult_z    = skip_im2col ? gemm_3d_depth : 1U;

    // Set dummy tensor shapes for the validation
    const TensorInfo dummy_input_info(TensorShape(4U, 4U * mult_y, 1U * mult_z), 1, data_type, input_info->quantization_info());
    const TensorInfo dummy_weights_info(TensorShape(4U, 4U), 1, data_type, weights_info->quantization_info());
    const TensorInfo dummy_output_info(TensorShape(4U, 4U, gemm_3d_depth), 1, data_type, input_info->quantization_info());

    return validate_mm(&dummy_input_info, &dummy_weights_info, nullptr, &dummy_output_info, act_info, false, gemm_3d_depth, skip_im2col);
}

void CpuGemmConvolution::configure(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst, const PadStrideInfo &conv_info, const WeightsInfo &weights_info,
                                   const Size2D &dilation, const ActivationLayerInfo &act_info, bool enable_fast_math, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_UNUSED(num_groups, weights_info);
    ARM_COMPUTE_ERROR_THROW_ON(CpuGemmConvolution::validate(src,
                                                            weights,
                                                            biases,
                                                            dst,
                                                            conv_info,
                                                            weights_info,
                                                            dilation,
                                                            act_info,
                                                            enable_fast_math,
                                                            num_groups));

    const DataType   data_type   = src->data_type();
    const DataLayout data_layout = src->data_layout();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        idx_kernels = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);

    const unsigned int kernel_width  = weights->dimension(idx_width);
    const unsigned int kernel_height = weights->dimension(idx_height);

    _is_prepared  = weights_info.retain_internal_weights();
    _is_quantized = is_data_type_quantized_asymmetric(src->data_type());
    _data_layout  = data_layout;
    _skip_im2col  = (data_layout == DataLayout::NHWC && kernel_width == 1 && kernel_height == 1 && conv_info.stride().first == 1 && conv_info.stride().second == 1);

    const ITensorInfo *gemm_input_to_use  = src;
    ITensorInfo       *gemm_output_to_use = dst;

    // Get convolved dimensions
    unsigned int conv_w = 0;
    unsigned int conv_h = 0;
    std::tie(conv_w, conv_h) = scaled_dimensions(src->dimension(idx_width),
                                                 src->dimension(idx_height),
                                                 kernel_width,
                                                 kernel_height,
                                                 conv_info,
                                                 dilation);
    ARM_COMPUTE_ERROR_ON_MSG((dst->dimension(idx_width) != conv_w) || (dst->dimension(idx_height) != conv_h),
                             "Output shape does not match the expected one");

    // Check if GEMM3D is supported
    if(data_layout == DataLayout::NHWC)
    {
        _skip_col2im = bool(validate_gemm3d(src, weights, act_info, conv_h, true));
        // If not supported, we need to perform im2col and col2im (or reshape layer)
        if(!_skip_col2im)
        {
            _skip_im2col = false;
        }
    }
    else
    {
        _skip_col2im = false;
    }

    // Get parameters from conv_info
    unsigned int stride_x = 0;
    unsigned int stride_y = 0;
    std::tie(stride_x, stride_y) = conv_info.stride();

    unsigned int mat_weights_cols = weights->dimension(idx_kernels);

    // _weights_reshaped will be auto configured in the kernel.
    // Just append biases and do not transpose 1xW as it will be reshaped in CpuGemm
    _weights_reshape_kernel = std::make_unique<kernels::CpuWeightsReshapeKernel>();
    _weights_reshape_kernel->configure(weights, nullptr, &_weights_reshaped);
    _weights_reshaped.set_quantization_info(weights->quantization_info());

    // Create tensor to store im2col reshaped inputs
    if(!_skip_im2col)
    {
        // Configure
        _im2col_kernel = std::make_unique<kernels::CpuIm2ColKernel>();
        _im2col_kernel->configure(src, &_im2col_output, Size2D(kernel_width, kernel_height), conv_info, false, dilation);

        // Update GEMM input
        gemm_input_to_use = &_im2col_output;
    }

    // Create temporary GEMM output tensor in case we cannot skip col2im
    const DataType output_data_type = data_type == DataType::BFLOAT16 ? DataType::F32 : data_type;
    if(!_skip_col2im)
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
    configure_mm(gemm_input_to_use, &_weights_reshaped, biases, gemm_output_to_use, act_info, enable_fast_math, gemm_3d_depth);

    if(!_skip_col2im && _data_layout == DataLayout::NCHW)
    {
        // Configure col2im
        _col2im_kernel = std::make_unique<kernels::CpuCol2ImKernel>();
        _col2im_kernel->configure(gemm_output_to_use, dst, Size2D(conv_w, conv_h));
    }
    else
    {
        // Configure reshape layer
        _reshape_kernel = std::make_unique<kernels::CpuReshapeKernel>();
        _reshape_kernel->configure(gemm_output_to_use, dst);
    }

    // Check if GEMM transforms weights
    // Modernise through COMPMID-4535
    bool gemm_trans_wei = _aux_mem[1].size > 0;                                            // Asm Pretranspose
    gemm_trans_wei      = _mm_gemm != nullptr ? _aux_mem[3].size > 0 : gemm_trans_wei;     // Tranpose RHS
    gemm_trans_wei      = _mm_gemmlowp != nullptr ? _aux_mem[5].size > 0 : gemm_trans_wei; // Transpose RHS

    // Check lifetime
    _aux_mem[Im2ColOutput]    = MemoryInfo(offset_int_vec(Im2ColOutput), MemoryLifetime::Temporary, _im2col_output.total_size());
    _aux_mem[WeightsReshaped] = MemoryInfo(offset_int_vec(WeightsReshaped), gemm_trans_wei ? MemoryLifetime::Prepare : MemoryLifetime::Persistent, _weights_reshaped.total_size());
    _aux_mem[GemmOutput]      = MemoryInfo(offset_int_vec(GemmOutput), MemoryLifetime::Temporary, _gemm_output.total_size());
}

Status CpuGemmConvolution::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const PadStrideInfo &conv_info,
                                    const WeightsInfo &weights_info, const Size2D &dilation, const ActivationLayerInfo &act_info, bool enable_fast_math, unsigned int num_groups)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights_info.are_reshaped(), "Weights already reshaped are not supported!");
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::BFLOAT16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM8_PER_CHANNEL, DataType::BFLOAT16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(src, weights);
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
    bool       skip_im2col  = (data_layout == DataLayout::NHWC && kernel_width == 1 && kernel_height == 1 && conv_info.stride().first == 1 && conv_info.stride().second == 1);

    // Get convolved dimensions
    unsigned int conv_w = 0;
    unsigned int conv_h = 0;

    std::tie(conv_w, conv_h) = scaled_dimensions(src->dimension(idx_width),
                                                 src->dimension(idx_height),
                                                 kernel_width,
                                                 kernel_height,
                                                 conv_info,
                                                 dilation);

    // Check if GEMM3D is supported
    bool skip_col2im = false;
    if(data_layout == DataLayout::NHWC)
    {
        skip_col2im = bool(validate_gemm3d(src, weights, act_info, conv_h, true));
        // If not supported, we need to perform im2col and col2im (or reshape layer)
        if(!skip_col2im)
        {
            skip_im2col = false;
        }
    }

    if(skip_col2im)
    {
        // If not supported, we need to perform im2col and col2im (or reshape layer)
        if(!bool(validate_gemm3d(src, weights, act_info, conv_h, skip_im2col)))
        {
            skip_im2col = false;
            skip_col2im = false;
        }
    }

    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_channel) != src->dimension(idx_channel));
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);

    // Validate biases
    if(biases != nullptr)
    {
        if(is_quantized)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::S32);
        }
        else if(is_bf16)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::F32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, biases);
        }
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != weights->dimension(idx_kernels));
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    }

    unsigned int mat_weights_cols = weights->dimension(idx_kernels);
    unsigned int mat_weights_rows = weights->dimension(idx_width) * weights->dimension(idx_height) * weights->dimension(idx_channel);

    weights_reshaped_info = TensorInfo(compute_weights_reshaped_shape(*weights, append_bias), 1, data_type);
    weights_reshaped_info.set_quantization_info(weights->quantization_info());
    weights_to_use = &weights_reshaped_info;

    if(!skip_im2col)
    {
        // Create tensor info for im2col reshaped inputs
        // For CPU, the batch size is on the fourth dimension
        TensorShape shape_im2col = src->tensor_shape();
        shape_im2col.set(0, mat_weights_rows);
        shape_im2col.set(1, conv_w * conv_h);
        shape_im2col.set(2, 1);

        im2col_reshaped_info = TensorInfo(shape_im2col, 1, data_type);
        im2col_reshaped_info.set_quantization_info(src->quantization_info());
        ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuIm2ColKernel::validate(src, &im2col_reshaped_info, Size2D(kernel_width, kernel_height), conv_info, append_bias, dilation));
        gemm_input_to_use = &im2col_reshaped_info;
    }

    // Create temporary GEMM output tensor in case we cannot skip col2im
    const DataType output_data_type = data_type == DataType::BFLOAT16 ? DataType::F32 : data_type;
    if(!skip_col2im)
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
    gemm_output_to_use = &info_gemm;
    ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(gemm_input_to_use, weights_to_use, biases, gemm_output_to_use, act_info, enable_fast_math, skip_col2im ? conv_h : 0, skip_im2col));

    // Validate Col2Im/ReshapeLayer
    if(!skip_col2im && (data_layout == DataLayout::NCHW))
    {
        ARM_COMPUTE_RETURN_ON_ERROR(kernels::CpuCol2ImKernel::validate(gemm_output_to_use, dst, Size2D(conv_w, conv_h)));
    }

    return Status{};
}

void CpuGemmConvolution::run(ITensorPack &tensors)
{
    prepare(tensors);

    auto src               = tensors.get_const_tensor(ACL_SRC_0);
    auto dst               = tensors.get_tensor(ACL_DST);
    auto gemm_input_to_use = src;

    CpuAuxTensorHandler im2col_output(offset_int_vec(Im2ColOutput), _im2col_output, tensors, false);
    CpuAuxTensorHandler gemm_output(offset_int_vec(GemmOutput), _gemm_output, tensors, false);
    CpuAuxTensorHandler reshaped_wei(offset_int_vec(WeightsReshaped), _weights_reshaped, tensors, false);

    bool out_has_padding = _skip_col2im && (dst->info()->padding().bottom != 0 || dst->info()->padding().top != 0);
    if(!_skip_im2col)
    {
        // Run input reshaping
        unsigned int y_dim = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);
        ITensorPack  pack =
        {
            { TensorType::ACL_SRC, src },
            { TensorType::ACL_DST, im2col_output.get() }
        };
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

    if(_skip_im2col)
    {
        gemm_output_to_use = &gemm3d;
    }
    if(_skip_col2im && !out_has_padding)
    {
        gemm_output_to_use = dst;
    }

    // Runs CpuGemm or CpuGemmLowpMatrixMultiplyCore functions
    ITensorPack pack_mm = tensors;
    pack_mm.add_const_tensor(TensorType::ACL_SRC_0, gemm_input_to_use);
    pack_mm.add_const_tensor(TensorType::ACL_SRC_1, reshaped_wei.get());
    pack_mm.add_tensor(TensorType::ACL_DST, gemm_output_to_use);
    if(_is_quantized)
    {
        // Run gemmlowp
        _mm_gemmlowp->run(pack_mm);
    }
    else
    {
        // Run gemm
        _mm_gemm->run(pack_mm);
    }

    // Reshape output matrix
    if(!_skip_col2im)
    {
        if(_data_layout == DataLayout::NCHW)
        {
            ITensorPack pack =
            {
                { TensorType::ACL_SRC, gemm_output.get() },
                { TensorType::ACL_DST, dst }
            };
            NEScheduler::get().schedule_op(_col2im_kernel.get(), Window::DimY, _col2im_kernel->window(), pack);
        }
        else
        {
            ITensorPack pack =
            {
                { TensorType::ACL_SRC, gemm_output_to_use },
                { TensorType::ACL_DST, dst }
            };
            NEScheduler::get().schedule_op(_reshape_kernel.get(), Window::DimY, _reshape_kernel->window(), pack);
        }
    }
    else if(out_has_padding)
    {
        ITensorPack pack =
        {
            { TensorType::ACL_SRC, gemm_output_to_use },
            { TensorType::ACL_DST, dst }
        };
        NEScheduler::get().schedule_op(_reshape_kernel.get(), Window::DimY, _reshape_kernel->window(), pack);
    }
}

void CpuGemmConvolution::prepare(ITensorPack &tensors)
{
    if(!_is_prepared)
    {
        // Run weights reshaping and mark original weights tensor as unused
        CpuAuxTensorHandler weights_reshaped(offset_int_vec(WeightsReshaped), _weights_reshaped, tensors);
        auto                weights = tensors.get_const_tensor(TensorType::ACL_SRC_1);
        ITensorPack         pack =
        {
            { TensorType::ACL_SRC, weights },
            { TensorType::ACL_DST, weights_reshaped.get() }
        };
        NEScheduler::get().schedule_op(_weights_reshape_kernel.get(), 3, _weights_reshape_kernel->window(), pack);
        weights->mark_as_unused();

        // Prepare GEMM
        ITensorPack gemm_pack = tensors;
        gemm_pack.add_const_tensor(TensorType::ACL_SRC_1, weights_reshaped.get());
        _is_quantized ? _mm_gemmlowp->prepare(gemm_pack) : _mm_gemm->prepare(gemm_pack);

        _is_prepared = true;
    }
}
experimental::MemoryRequirements CpuGemmConvolution::workspace() const
{
    return _aux_mem;
}
} // namespace cpu
} // namespace arm_compute