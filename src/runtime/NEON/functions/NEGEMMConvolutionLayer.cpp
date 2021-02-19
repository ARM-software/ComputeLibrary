/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEGEMMConvolutionLayer.h"

#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/core/NEON/kernels/NECol2ImKernel.h"
#include "src/core/NEON/kernels/NEConvertQuantizedSignednessKernel.h"
#include "src/core/NEON/kernels/NEGEMMInterleave4x4Kernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpMatrixMultiplyKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpOffsetContributionKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpOffsetContributionOutputStageKernel.h"
#include "src/core/NEON/kernels/NEGEMMLowpReductionKernel.h"
#include "src/core/NEON/kernels/NEGEMMMatrixAdditionKernel.h"
#include "src/core/NEON/kernels/NEGEMMMatrixMultiplyKernel.h"
#include "src/core/NEON/kernels/NEGEMMTranspose1xWKernel.h"
#include "src/core/NEON/kernels/NEGEMMTranspose1xWKernel.h"
#include "src/core/NEON/kernels/NEIm2ColKernel.h"
#include "src/core/NEON/kernels/NEWeightsReshapeKernel.h"

#include <set>
#include <tuple>

namespace arm_compute
{
using namespace arm_compute::misc::shape_calculator;

NEConvolutionLayerReshapeWeights::~NEConvolutionLayerReshapeWeights() = default;
NEConvolutionLayerReshapeWeights::NEConvolutionLayerReshapeWeights()
    : _weights_reshape_kernel()
{
}

void NEConvolutionLayerReshapeWeights::configure(const ITensor *weights, const ITensor *biases, ITensor *output)
{
    // Perform validation step
    ARM_COMPUTE_ERROR_ON_NULLPTR(weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(NEConvolutionLayerReshapeWeights::validate(weights->info(),
                                                                          (biases != nullptr) ? biases->info() : nullptr,
                                                                          output->info()));
    const bool     append_biases = (biases != nullptr) && !is_data_type_quantized_asymmetric(weights->info()->data_type());
    const ITensor *biases_to_use = (append_biases) ? biases : nullptr;

    _weights_reshape_kernel = std::make_unique<NEWeightsReshapeKernel>();
    _weights_reshape_kernel->configure(weights, biases_to_use, output);

    output->info()->set_quantization_info(weights->info()->quantization_info());
}

Status NEConvolutionLayerReshapeWeights::validate(const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(weights);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1,
                                                         DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM8_PER_CHANNEL,
                                                         DataType::BFLOAT16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);

    if(biases != nullptr)
    {
        const int idx_kernels = get_data_layout_dimension_index(weights->data_layout(), DataLayoutDimension::BATCHES);
        ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_asymmetric(weights->data_type()));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != weights->dimension(idx_kernels));
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    }

    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, output);

        NEWeightsReshapeKernel::validate(weights, biases, output);
    }

    return Status{};
}

void NEConvolutionLayerReshapeWeights::run()
{
    NEScheduler::get().schedule(_weights_reshape_kernel.get(), 3);
}

NEGEMMConvolutionLayer::~NEGEMMConvolutionLayer() = default;

NEGEMMConvolutionLayer::NEGEMMConvolutionLayer(const std::shared_ptr<IMemoryManager> &memory_manager, IWeightsManager *weights_manager)
    : _memory_group(memory_manager), _weights_manager(weights_manager), _reshape_weights(), _reshape_weights_managed(), _im2col_kernel(), _mm_gemm(memory_manager), _mm_gemmlowp(memory_manager),
      _col2im_kernel(), _reshape_layer(), _original_weights(nullptr), _original_output(nullptr), _im2col_output(), _weights_reshaped(), _gemm_output(), _gemm_output_3d(), _tmp_output(),
      _data_layout(DataLayout::NCHW), _skip_im2col(false), _skip_col2im(false), _is_quantized(false), _is_prepared(false)
{
}

void NEGEMMConvolutionLayer::configure_mm(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const ActivationLayerInfo &act_info, int gemm_3d_depth)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights);
    ARM_COMPUTE_ERROR_THROW_ON(validate_mm(input->info(), weights->info(), biases == nullptr ? nullptr : biases->info(), output == nullptr ? nullptr : output->info(),
                                           act_info, gemm_3d_depth, _skip_im2col));

    // Create GEMMInfo structure
    const GEMMInfo &gemm_info = GEMMInfo(false, false, true /* Reshape weights only for the first run */,
                                         gemm_3d_depth, _skip_im2col /* Reinterpret the input as 3D if im2col is skipped */,
                                         false, GEMMLowpOutputStageInfo(), false, false, act_info);

    // Supported activations in GEMM
    const std::set<ActivationLayerInfo::ActivationFunction> supported_acts = { ActivationLayerInfo::ActivationFunction::RELU,
                                                                               ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
                                                                               ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU
                                                                             };

    if(_is_quantized)
    {
        // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
        // Extract and negate input and weights offset
        const QuantizationInfo        iqinfo    = input->info()->quantization_info();
        const QuantizationInfo        wqinfo    = weights->info()->quantization_info();
        const QuantizationInfo        oqinfo    = (output->info()->total_size() == 0) ? iqinfo : output->info()->quantization_info();
        const UniformQuantizationInfo uiqinfo   = iqinfo.uniform();
        const UniformQuantizationInfo uoqinfo   = oqinfo.uniform();
        const DataType                data_type = input->info()->data_type();

        input->info()->set_quantization_info(QuantizationInfo(uiqinfo.scale, -uiqinfo.offset));
        if(!is_data_type_quantized_per_channel(weights->info()->data_type()))
        {
            const UniformQuantizationInfo uwqinfo = wqinfo.uniform();
            weights->info()->set_quantization_info(QuantizationInfo(uwqinfo.scale, -uwqinfo.offset));
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
        output_info.is_quantized_per_channel = (weights->info()->data_type() == DataType::QSYMM8_PER_CHANNEL);
        quantization::calculate_quantized_multipliers(iqinfo, wqinfo, oqinfo, output_info);

        _mm_gemmlowp.configure(input, weights, biases, output, GEMMInfo(false, false, true, gemm_3d_depth, _skip_im2col, false, output_info, false, false, act_info));

        // Revert back QuantizatioInfo as input and weights could be used in other convolution layers
        input->info()->set_quantization_info(iqinfo);
        weights->info()->set_quantization_info(wqinfo);
    }
    else
    {
        // Configure matrix multiply function
        _mm_gemm.configure(input, weights, biases, output, 1.0f, 0.0f, gemm_info);
    }
}

Status NEGEMMConvolutionLayer::validate_mm(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                                           const ActivationLayerInfo &act_info, int gemm_3d_depth, bool skip_im2col)
{
    const DataType data_type             = input->data_type();
    const bool     is_quantized          = is_data_type_quantized_asymmetric(data_type);
    const bool     is_activation_enabled = act_info.enabled();

    // Create GEMMInfo structure
    const GEMMInfo gemm_info = GEMMInfo(false, false, true /* Reshape weights only for the first run */,
                                        gemm_3d_depth, skip_im2col /* Reinterpret the input as 3D if im2col is skipped */,
                                        false, GEMMLowpOutputStageInfo(), false, false, act_info);

    if(is_quantized)
    {
        // Since we need negative offsets for computing convolution, we need to change QuantizationInfo()
        // Extract and negate input and weights offset
        const QuantizationInfo       &iqinfo  = input->quantization_info();
        const QuantizationInfo       &wqinfo  = weights->quantization_info();
        const QuantizationInfo       &oqinfo  = (output->total_size() == 0) ? iqinfo : output->quantization_info();
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
        std::unique_ptr<ITensorInfo> input_qa   = input->clone();
        std::unique_ptr<ITensorInfo> weights_qa = weights->clone();
        input_qa->set_quantization_info(QuantizationInfo(iqinfo.uniform().scale, -iqinfo.uniform().offset));
        weights_qa->set_quantization_info(QuantizationInfo(wqinfo.uniform().scale, -wqinfo.uniform().offset));
        return NEGEMMLowpMatrixMultiplyCore::validate(input_qa.get(), weights_qa.get(), biases, output, GEMMInfo(false, false, true, gemm_3d_depth, skip_im2col, false, output_info, false, false, act_info));
    }
    else
    {
        // Perform validation step on Matrix multiply function
        return NEGEMM::validate(input, weights, nullptr, output, 1.0f, 0.0f, gemm_info);
    }
}

Status NEGEMMConvolutionLayer::validate_gemm3d(const ITensorInfo *input_info, const ITensorInfo *weights_info, const ActivationLayerInfo &act_info, int gemm_3d_depth, bool skip_im2col)
{
    const DataType     data_type = input_info->data_type();
    const unsigned int mult_y    = skip_im2col ? 1U : gemm_3d_depth;
    const unsigned int mult_z    = skip_im2col ? gemm_3d_depth : 1U;

    // Set dummy tensor shapes for the validation
    const TensorInfo dummy_input_info(TensorShape(4U, 4U * mult_y, 1U * mult_z), 1, data_type, input_info->quantization_info());
    const TensorInfo dummy_weights_info(TensorShape(4U, 4U), 1, data_type, weights_info->quantization_info());
    const TensorInfo dummy_output_info(TensorShape(4U, 4U, gemm_3d_depth), 1, data_type, input_info->quantization_info());

    return validate_mm(&dummy_input_info, &dummy_weights_info, nullptr, &dummy_output_info, act_info, gemm_3d_depth, skip_im2col);
}

void NEGEMMConvolutionLayer::configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info,
                                       const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_UNUSED(num_groups, weights_info);
    ARM_COMPUTE_ERROR_THROW_ON(NEGEMMConvolutionLayer::validate(input->info(),
                                                                weights->info(),
                                                                biases != nullptr ? biases->info() : nullptr,
                                                                output->info(),
                                                                conv_info,
                                                                weights_info,
                                                                dilation,
                                                                act_info,
                                                                num_groups));

    const DataType   data_type   = input->info()->data_type();
    const DataLayout data_layout = input->info()->data_layout();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        idx_kernels = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);

    const unsigned int kernel_width  = weights->info()->dimension(idx_width);
    const unsigned int kernel_height = weights->info()->dimension(idx_height);

    _is_prepared      = weights_info.retain_internal_weights();
    _original_weights = weights;
    _original_output  = output;
    _is_quantized     = is_data_type_quantized_asymmetric(input->info()->data_type());
    _data_layout      = data_layout;
    _skip_im2col      = (data_layout == DataLayout::NHWC && kernel_width == 1 && kernel_height == 1 && conv_info.stride().first == 1 && conv_info.stride().second == 1);

    const ITensor *gemm_input_to_use  = input;
    ITensor       *gemm_output_to_use = output;

    // Get convolved dimensions
    unsigned int conv_w = 0;
    unsigned int conv_h = 0;
    std::tie(conv_w, conv_h) = scaled_dimensions(input->info()->dimension(idx_width),
                                                 input->info()->dimension(idx_height),
                                                 kernel_width,
                                                 kernel_height,
                                                 conv_info,
                                                 dilation);

    // Check if GEMM3D is supported
    if(data_layout == DataLayout::NHWC)
    {
        _skip_col2im = bool(validate_gemm3d(input->info(), weights->info(), act_info, conv_h, true));
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

    unsigned int mat_weights_cols = weights->info()->dimension(idx_kernels);

    // _weights_reshaped will be auto configured in the kernel.
    // Just append biases and do not transpose 1xW as it will be reshaped in NEGEMM
    const ITensor *weights_to_use = weights;

    if(_weights_manager && _weights_manager->are_weights_managed(weights))
    {
        _reshape_weights_managed.configure(weights, nullptr);
        weights_to_use = _weights_manager->acquire(weights, &_reshape_weights_managed);
    }
    else
    {
        _reshape_weights.configure(weights, nullptr, &_weights_reshaped);
        weights_to_use = &_weights_reshaped;
    }

    // Create tensor to store im2col reshaped inputs
    if(!_skip_im2col)
    {
        _memory_group.manage(&_im2col_output);

        // Configure
        _im2col_kernel = std::make_unique<NEIm2ColKernel>();
        _im2col_kernel->configure(input, &_im2col_output, Size2D(kernel_width, kernel_height), conv_info, false, dilation);

        // Update GEMM input
        gemm_input_to_use = &_im2col_output;
    }

    // Create temporary GEMM output tensor in case we cannot skip col2im
    const DataType output_data_type = data_type == DataType::BFLOAT16 ? DataType::F32 : data_type;
    if(!_skip_col2im)
    {
        TensorShape shape_gemm;

        // Calculate GEMM output shape
        shape_gemm = _im2col_output.info()->tensor_shape();
        shape_gemm.set(0, mat_weights_cols);
        shape_gemm.set(1, conv_w * conv_h);

        // FIXME: input->clone() doesn't work with subtensors for grouped convolutions.
        TensorInfo info_gemm(shape_gemm, 1, output_data_type);
        info_gemm.set_quantization_info(output->info()->quantization_info()).set_data_layout(input->info()->data_layout());
        _gemm_output.allocator()->init(info_gemm);
        _gemm_output_3d.allocator()->init(info_gemm);
        _memory_group.manage(&_gemm_output);

        // Update GEMM output
        gemm_output_to_use = &_gemm_output;
    }
    else
    {
        TensorInfo out_info{ *output->info() };
        out_info.set_data_type(output_data_type).set_data_layout(input->info()->data_layout());
        _gemm_output.allocator()->init(out_info);
        _gemm_output_3d.allocator()->init(out_info);
        _memory_group.manage(&_gemm_output);

        // Update GEMM output
        gemm_output_to_use = &_gemm_output_3d;
    }

    // Configure GEMM
    // In case we need to skip col2im, GEMM3D (gemm_3d_depth != 0) must be called in order to avoid reshaping the output matrix
    const unsigned int gemm_3d_depth = _skip_col2im ? conv_h : 0;
    configure_mm(gemm_input_to_use, weights_to_use, biases, gemm_output_to_use, act_info, gemm_3d_depth);

    if(!_skip_im2col)
    {
        _im2col_output.allocator()->allocate();
    }

    if(!_skip_col2im)
    {
        if(_data_layout == DataLayout::NCHW)
        {
            // Configure col2im
            _col2im_kernel = std::make_unique<NECol2ImKernel>();
            _col2im_kernel->configure(gemm_output_to_use, output, Size2D(conv_w, conv_h));
        }
        else
        {
            // Configure reshape layer
            _reshape_layer.configure(gemm_output_to_use, output);
        }
    }
    else
    {
        // Configure reshape layer
        _reshape_layer.configure(gemm_output_to_use, output);
    }

    if(_is_quantized && !_skip_col2im)
    {
        _tmp_output.allocator()->allocate();
    }

    _gemm_output.allocator()->allocate();

    ARM_COMPUTE_ERROR_ON_MSG((output->info()->dimension(idx_width) != conv_w) || (output->info()->dimension(idx_height) != conv_h),
                             "Output shape does not match the expected one");
}

Status NEGEMMConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                        const WeightsInfo &weights_info, const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights_info.are_reshaped(), "Weights already reshaped are not supported!");
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::BFLOAT16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM8_PER_CHANNEL, DataType::BFLOAT16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(num_groups > 1, "Grouping (num_groups != 1) is not supported on NEON");

    const DataLayout data_layout = input->data_layout();
    const DataType   data_type   = input->data_type();
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
    const ITensorInfo *gemm_input_to_use  = input;
    const ITensorInfo *gemm_output_to_use = output;
    const ITensorInfo *weights_to_use     = weights;

    const bool append_bias  = false;
    const bool is_quantized = is_data_type_quantized_asymmetric(data_type);
    const bool is_bf16      = data_type == DataType::BFLOAT16;
    bool       skip_im2col  = (data_layout == DataLayout::NHWC && kernel_width == 1 && kernel_height == 1 && conv_info.stride().first == 1 && conv_info.stride().second == 1);

    // Get convolved dimensions
    unsigned int conv_w = 0;
    unsigned int conv_h = 0;

    std::tie(conv_w, conv_h) = scaled_dimensions(input->dimension(idx_width),
                                                 input->dimension(idx_height),
                                                 kernel_width,
                                                 kernel_height,
                                                 conv_info,
                                                 dilation);

    // Check if GEMM3D is supported
    bool skip_col2im = false;
    if(data_layout == DataLayout::NHWC)
    {
        skip_col2im = bool(validate_gemm3d(input, weights, act_info, conv_h, true));
        // If not supported, we need to perform im2col and col2im (or reshape layer)
        if(!skip_col2im)
        {
            skip_im2col = false;
        }
    }

    if(skip_col2im)
    {
        // If not supported, we need to perform im2col and col2im (or reshape layer)
        if(!bool(validate_gemm3d(input, weights, act_info, conv_h, skip_im2col)))
        {
            skip_im2col = false;
            skip_col2im = false;
        }
    }

    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_channel) != input->dimension(idx_channel));
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
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        }
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != weights->dimension(idx_kernels));
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    }

    unsigned int mat_weights_cols = weights->dimension(idx_kernels);
    unsigned int mat_weights_rows = weights->dimension(idx_width) * weights->dimension(idx_height) * weights->dimension(idx_channel);

    // Output tensor auto inizialization if not yet initialized
    ARM_COMPUTE_RETURN_ON_ERROR(NEConvolutionLayerReshapeWeights::validate(weights, nullptr, nullptr));
    weights_reshaped_info = TensorInfo(compute_weights_reshaped_shape(*weights, append_bias), 1, data_type);
    weights_reshaped_info.set_quantization_info(weights->quantization_info());
    weights_to_use = &weights_reshaped_info;

    if(!skip_im2col)
    {
        // Create tensor info for im2col reshaped inputs
        // For NEON the batch size is on the fourth dimension
        // TODO (giaiod01): Auto-initialize the output shape of im2col COMPMID-1482
        TensorShape shape_im2col = input->tensor_shape();
        shape_im2col.set(0, mat_weights_rows);
        shape_im2col.set(1, conv_w * conv_h);
        shape_im2col.set(2, 1);

        im2col_reshaped_info = TensorInfo(shape_im2col, 1, data_type);
        im2col_reshaped_info.set_quantization_info(input->quantization_info());

        ARM_COMPUTE_RETURN_ON_ERROR(NEIm2ColKernel::validate(input, &im2col_reshaped_info, Size2D(kernel_width, kernel_height), conv_info, append_bias, dilation));
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
        info_gemm = TensorInfo(output->tensor_shape(), 1, output_data_type);
    }
    info_gemm.set_quantization_info(output->quantization_info()).set_data_layout(input->data_layout());
    gemm_output_to_use = &info_gemm;
    ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(gemm_input_to_use, weights_to_use, biases, gemm_output_to_use, act_info, skip_col2im ? conv_h : 0, skip_im2col));

    // Validate Col2Im/ReshapeLayer
    if(!skip_col2im && (data_layout == DataLayout::NCHW))
    {
        ARM_COMPUTE_RETURN_ON_ERROR(NECol2ImKernel::validate(gemm_output_to_use, output, Size2D(conv_w, conv_h)));
    }

    return Status{};
}

void NEGEMMConvolutionLayer::run()
{
    prepare();

    MemoryGroupResourceScope scope_mg(_memory_group);

    bool out_has_padding = _skip_col2im && (_original_output->info()->padding().bottom != 0 || _original_output->info()->padding().top != 0);

    if(!_skip_im2col)
    {
        // Run input reshaping
        unsigned int y_dim = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);
        NEScheduler::get().schedule(_im2col_kernel.get(), y_dim);
    }

    // Handle the case where output has top/bottom padding
    const ITensor *out_to_use = out_has_padding ? &_gemm_output : _original_output;
    _gemm_output_3d.info()->extend_padding(out_to_use->info()->padding());
    _gemm_output_3d.allocator()->import_memory(out_to_use->buffer());

    // Runs NEGEMM or NEGEMMLowpMatrixMultiplyCore functions
    if(_is_quantized)
    {
        // Run gemmlowp
        _mm_gemmlowp.run();
    }
    else
    {
        // Run gemm
        _mm_gemm.run();
    }

    // Reshape output matrix
    if(!_skip_col2im)
    {
        if(_data_layout == DataLayout::NCHW)
        {
            NEScheduler::get().schedule(_col2im_kernel.get(), Window::DimY);
        }
        else
        {
            _reshape_layer.run();
        }
    }
    else if(out_has_padding)
    {
        _reshape_layer.run();
    }

    _gemm_output_3d.allocator()->free();
}

void NEGEMMConvolutionLayer::prepare()
{
    if(!_is_prepared)
    {
        if(_weights_manager && _weights_manager->are_weights_managed(_original_weights))
        {
            _weights_manager->run(_original_weights, &_reshape_weights_managed);
        }
        else
        {
            // Run weights reshaping and mark original weights tensor as unused
            _weights_reshaped.allocator()->allocate();
            _reshape_weights.run();
            _original_weights->mark_as_unused();
        }

        // Prepare GEMM
        _is_quantized ? _mm_gemmlowp.prepare() : _mm_gemm.prepare();
        if(!_weights_reshaped.is_used())
        {
            _weights_reshaped.allocator()->free();
        }

        _is_prepared = true;
    }
}
} // namespace arm_compute
