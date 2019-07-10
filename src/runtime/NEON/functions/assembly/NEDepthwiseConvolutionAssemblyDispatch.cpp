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

#include "arm_compute/runtime/NEON/functions/assembly/NEDepthwiseConvolutionAssemblyDispatch.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/kernels/assembly/NEDepthwiseConvolutionAssemblyKernelWrapper.h"
#include "arm_compute/core/NEON/kernels/convolution/depthwise/depthwise_dilated.hpp"
#include "arm_compute/core/NEON/kernels/convolution/depthwise/depthwise_quantized_dilated.hpp"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/InfoHelpers.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

#include <set>

namespace arm_compute
{
namespace
{
std::unique_ptr<depthwise::IDepthwiseConvolution> get_qasymm8_convolver(int kernel_size, int stride_x,
                                                                        int n_batches, int in_rows, int in_cols, int n_channels,
                                                                        int dilation_factor, neon_convolution_kernels::ActivationFunction activation,
                                                                        const qasymm8::QAsymm8Params &wqinfo, const qasymm8::QAsymm8Params &iqinfo, const qasymm8::QAsymm8Params &oqinfo,
                                                                        const qasymm8::QAsymm8RescaleParams &rescale_params,
                                                                        int padding_top, int padding_left, int padding_bottom, int padding_right)
{
    switch(kernel_size)
    {
        case 3:
        {
            switch(stride_x)
            {
                case 1:
                    return arm_compute::support::cpp14::make_unique<depthwise::QAsymm8DilatedDepthwiseConvolution<2, 2, 3, 3, 1, 1>>(
                               n_batches, in_rows, in_cols, n_channels, dilation_factor, activation, wqinfo, iqinfo, oqinfo, rescale_params, padding_top, padding_left, padding_bottom, padding_right);
                case 2:
                    return arm_compute::support::cpp14::make_unique<depthwise::QAsymm8DilatedDepthwiseConvolution<2, 2, 3, 3, 2, 2>>(
                               n_batches, in_rows, in_cols, n_channels, dilation_factor, activation, wqinfo, iqinfo, oqinfo, rescale_params, padding_top, padding_left, padding_bottom, padding_right);
                default:
                    return nullptr;
            }
        }
        case 5:
        {
            switch(stride_x)
            {
                case 1:
                    return arm_compute::support::cpp14::make_unique<depthwise::QAsymm8DilatedDepthwiseConvolution<2, 2, 5, 5, 1, 1>>(
                               n_batches, in_rows, in_cols, n_channels, dilation_factor, activation, wqinfo, iqinfo, oqinfo, rescale_params, padding_top, padding_left, padding_bottom, padding_right);
                case 2:
                    return arm_compute::support::cpp14::make_unique<depthwise::QAsymm8DilatedDepthwiseConvolution<2, 2, 5, 5, 2, 2>>(
                               n_batches, in_rows, in_cols, n_channels, dilation_factor, activation, wqinfo, iqinfo, oqinfo, rescale_params, padding_top, padding_left, padding_bottom, padding_right);
                default:
                    return nullptr;
            }
        }
        default:
            return nullptr;
    }
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
std::unique_ptr<depthwise::IDepthwiseConvolution> get_fp16_convolver(int kernel_size, int stride_x,
                                                                     int n_batches, int in_rows, int in_cols, int n_channels,
                                                                     int dilation_factor, neon_convolution_kernels::ActivationFunction activation,
                                                                     int padding_top, int padding_left, int padding_bottom, int padding_right)
{
    switch(kernel_size)
    {
        case 3:
        {
            switch(stride_x)
            {
                case 1:
                    return arm_compute::support::cpp14::make_unique<depthwise::DilatedDepthwiseConvolution<3, 3, 3, 3, 1, 1, float16_t, float16_t, float16_t>>(
                               n_batches, in_rows, in_cols, n_channels, dilation_factor, activation, padding_top, padding_left, padding_bottom, padding_right);
                case 2:
                    return arm_compute::support::cpp14::make_unique<depthwise::DilatedDepthwiseConvolution<3, 3, 3, 3, 2, 2, float16_t, float16_t, float16_t>>(
                               n_batches, in_rows, in_cols, n_channels, dilation_factor, activation, padding_top, padding_left, padding_bottom, padding_right);
                default:
                    return nullptr;
            }
        }
        case 5:
        {
            switch(stride_x)
            {
                case 1:
                    return arm_compute::support::cpp14::make_unique<depthwise::DilatedDepthwiseConvolution<3, 3, 5, 5, 1, 1, float16_t, float16_t, float16_t>>(
                               n_batches, in_rows, in_cols, n_channels, dilation_factor, activation, padding_top, padding_left, padding_bottom, padding_right);
                case 2:
                    return arm_compute::support::cpp14::make_unique<depthwise::DilatedDepthwiseConvolution<3, 3, 5, 5, 2, 2, float16_t, float16_t, float16_t>>(
                               n_batches, in_rows, in_cols, n_channels, dilation_factor, activation, padding_top, padding_left, padding_bottom, padding_right);
                default:
                    return nullptr;
            }
        }
        default:
            return nullptr;
    }
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

std::unique_ptr<depthwise::IDepthwiseConvolution> get_fp32_convolver(int kernel_size, int stride_x,
                                                                     int n_batches, int in_rows, int in_cols, int n_channels,
                                                                     int dilation_factor, neon_convolution_kernels::ActivationFunction activation,
                                                                     int padding_top, int padding_left, int padding_bottom, int padding_right)
{
    switch(kernel_size)
    {
        case 3:
        {
            switch(stride_x)
            {
                case 1:
                    return arm_compute::support::cpp14::make_unique<depthwise::DilatedDepthwiseConvolution<4, 4, 3, 3, 1, 1, float, float, float>>(
                               n_batches, in_rows, in_cols, n_channels, dilation_factor, activation, padding_top, padding_left, padding_bottom, padding_right);
                case 2:
                    return arm_compute::support::cpp14::make_unique<depthwise::DilatedDepthwiseConvolution<3, 3, 3, 3, 2, 2, float, float, float>>(
                               n_batches, in_rows, in_cols, n_channels, dilation_factor, activation, padding_top, padding_left, padding_bottom, padding_right);
                default:
                    return nullptr;
            }
        }
        case 5:
        {
            switch(stride_x)
            {
                case 1:
                    return arm_compute::support::cpp14::make_unique<depthwise::DilatedDepthwiseConvolution<4, 4, 5, 5, 1, 1, float, float, float>>(
                               n_batches, in_rows, in_cols, n_channels, dilation_factor, activation, padding_top, padding_left, padding_bottom, padding_right);
                case 2:
                    return arm_compute::support::cpp14::make_unique<depthwise::DilatedDepthwiseConvolution<3, 3, 5, 5, 2, 2, float, float, float>>(
                               n_batches, in_rows, in_cols, n_channels, dilation_factor, activation, padding_top, padding_left, padding_bottom, padding_right);
                default:
                    return nullptr;
            }
        }
        default:
            return nullptr;
    }
}

std::unique_ptr<depthwise::IDepthwiseConvolution> create_convolver(const ITensor      *input,
                                                                   const ITensor      *weights,
                                                                   ITensor            *output,
                                                                   PadStrideInfo       conv_info,
                                                                   ActivationLayerInfo act_info,
                                                                   const Size2D       &dilation)
{
    ARM_COMPUTE_UNUSED(dilation);
    const DataType    data_type = input->info()->data_type();
    const TensorShape shape     = input->info()->tensor_shape();

    const int n_batches       = shape[3];
    const int in_rows         = shape.z();
    const int in_cols         = shape.y();
    const int n_channels      = shape.x();
    const int dilation_factor = dilation.x();
    const int padding_top     = conv_info.pad_top();
    const int padding_left    = conv_info.pad_left();
    const int padding_bottom  = conv_info.pad_bottom();
    const int padding_right   = conv_info.pad_right();

    const unsigned int stride_x    = conv_info.stride().first;
    const unsigned int kernel_size = weights->info()->tensor_shape().y();

    // Map activation function
    neon_convolution_kernels::ActivationFunction activation = neon_convolution_kernels::ActivationFunction::None;
    if(arm_compute::utils::info_helpers::is_relu(act_info))
    {
        activation = neon_convolution_kernels::ActivationFunction::ReLU;
    }
    else if(arm_compute::utils::info_helpers::is_relu6(act_info))
    {
        activation = neon_convolution_kernels::ActivationFunction::ReLU6;
    }

    // Create quantized convolver
    if(data_type == DataType::QASYMM8)
    {
        const UniformQuantizationInfo input_qinfo   = input->info()->quantization_info().uniform();
        const UniformQuantizationInfo weights_qinfo = weights->info()->quantization_info().uniform();
        const UniformQuantizationInfo output_qinfo  = output->info()->quantization_info().uniform();

        // Check that quantization info are in the range [0, 255]
        ARM_COMPUTE_ERROR_ON(input_qinfo.offset < 0 || input_qinfo.offset > 255);
        ARM_COMPUTE_ERROR_ON(weights_qinfo.offset < 0 || weights_qinfo.offset > 255);
        ARM_COMPUTE_ERROR_ON(output_qinfo.offset < 0 || output_qinfo.offset > 255);
        const qasymm8::QAsymm8Params iqinfo{ static_cast<uint8_t>(input_qinfo.offset), input_qinfo.scale };
        const qasymm8::QAsymm8Params wqinfo{ static_cast<uint8_t>(weights_qinfo.offset), weights_qinfo.scale };
        const qasymm8::QAsymm8Params oqinfo{ static_cast<uint8_t>(output_qinfo.offset), output_qinfo.scale };

        // Calculate rescale parameters
        const float fmultipler  = iqinfo.scale * wqinfo.scale / oqinfo.scale;
        int         qmultiplier = 0;
        int         qshift      = 0;
        quantization::calculate_quantized_multiplier_less_than_one(fmultipler, &qmultiplier, &qshift);
        qasymm8::QAsymm8RescaleParams rescale_params(qshift, qmultiplier, fmultipler);

        return get_qasymm8_convolver(kernel_size, stride_x, n_batches, in_rows, in_cols, n_channels, dilation_factor, activation,
                                     wqinfo, iqinfo, oqinfo, rescale_params, padding_top, padding_left, padding_bottom, padding_right);
    }
    else
    {
        // Create float convolver
        switch(data_type)
        {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            case DataType::F16:
            {
                return get_fp16_convolver(kernel_size, stride_x, n_batches, in_rows, in_cols, n_channels, dilation_factor, activation, padding_top, padding_left, padding_bottom, padding_right);
            }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            case DataType::F32:
            {
                return get_fp32_convolver(kernel_size, stride_x, n_batches, in_rows, in_cols, n_channels, dilation_factor, activation, padding_top, padding_left, padding_bottom, padding_right);
            }
            default:
                return nullptr;
        }
    }
}
} // namespace

struct NEDepthwiseConvolutionAssemblyDispatch::LocalImpl
{
    std::unique_ptr<depthwise::IDepthwiseConvolution> _dwc_assembly_kernel{ nullptr };
    NEDepthwiseConvolutionAssemblyKernelWrapper       _dwc_acl_kernel{};
};

#ifndef DOXYGEN_SKIP_THIS
NEDepthwiseConvolutionAssemblyDispatch::NEDepthwiseConvolutionAssemblyDispatch(std::shared_ptr<arm_compute::IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _input(nullptr), _weights(nullptr), _bias(nullptr), _output(nullptr), _packed_weights(), _workspace(), _is_prepared(false),
      _pImpl(support::cpp14::make_unique<LocalImpl>())
{
}
#endif /* DOXYGEN_SKIP_THIS */

NEDepthwiseConvolutionAssemblyDispatch::~NEDepthwiseConvolutionAssemblyDispatch() = default;

void NEDepthwiseConvolutionAssemblyDispatch::configure(const ITensor             *input,
                                                       const ITensor             *weights,
                                                       const ITensor             *bias,
                                                       ITensor                   *output,
                                                       const PadStrideInfo       &conv_info,
                                                       unsigned int               depth_multiplier,
                                                       const ActivationLayerInfo &act_info,
                                                       const Size2D              &dilation)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_UNUSED(depth_multiplier);
    ARM_COMPUTE_ERROR_THROW_ON(NEDepthwiseConvolutionAssemblyDispatch::validate(input->info(),
                                                                                weights->info(),
                                                                                bias != nullptr ? bias->info() : nullptr,
                                                                                output->info(),
                                                                                conv_info,
                                                                                depth_multiplier,
                                                                                act_info,
                                                                                dilation));

    // Output auto inizialitation if not yet initialized
    const TensorShape output_shape = misc::shape_calculator::compute_depthwise_convolution_shape(*input->info(), *weights->info(), conv_info, depth_multiplier, dilation);
    auto_init_if_empty(*output->info(), input->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(output_shape).set_quantization_info(output->info()->quantization_info()));

    _input       = input;
    _weights     = weights;
    _bias        = bias;
    _output      = output;
    _is_prepared = false;

    // Create convolver
    _pImpl->_dwc_assembly_kernel = create_convolver(input, weights, output, conv_info, act_info, dilation);
    ARM_COMPUTE_ERROR_ON(_pImpl->_dwc_assembly_kernel == nullptr);

    // Create assembly kernel wrapper
    _pImpl->_dwc_acl_kernel.configure(_pImpl->_dwc_assembly_kernel.get());

    constexpr size_t alignment = 128;

    // Create workspace
    const unsigned int num_threads    = NEScheduler::get().num_threads();
    const size_t       workspace_size = _pImpl->_dwc_assembly_kernel->get_working_space_size(num_threads);
    ARM_COMPUTE_ERROR_ON_MSG(workspace_size == 0, "Workspace size cannot be 0 !");
    _workspace.allocator()->init(TensorInfo(TensorShape{ workspace_size }, 1, DataType::S8), alignment);
    _memory_group.manage(&_workspace);
    _workspace.allocator()->allocate();

    // Create packing tensor
    const size_t pack_tensor_size = _pImpl->_dwc_assembly_kernel->get_packed_params_size();
    ARM_COMPUTE_ERROR_ON_MSG(pack_tensor_size == 0, "Pack tensor size cannot be 0 !");
    _packed_weights.allocator()->init(TensorInfo(TensorShape{ pack_tensor_size }, 1, DataType::S8), alignment);
}

Status NEDepthwiseConvolutionAssemblyDispatch::validate(const ITensorInfo         *input,
                                                        const ITensorInfo         *weights,
                                                        const ITensorInfo         *bias,
                                                        const ITensorInfo         *output,
                                                        const PadStrideInfo       &conv_info,
                                                        unsigned int               depth_multiplier,
                                                        const ActivationLayerInfo &act_info,
                                                        const Size2D              &dilation)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, weights);

    // Validate convolver
    ARM_COMPUTE_RETURN_ERROR_ON(!is_optimized_supported(input, weights, conv_info, depth_multiplier, dilation));

    // Validate activation
    const bool is_relu  = arm_compute::utils::info_helpers::is_relu(act_info);
    const bool is_relu6 = arm_compute::utils::info_helpers::is_relu6(act_info);
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled() && !(is_relu || is_relu6));

    // Check bias
    if(bias != nullptr)
    {
        unsigned int channel_idx = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL);
        ARM_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(bias->dimension(0) != weights->dimension(channel_idx));
    }

    // Check output
    if(output->total_size() != 0)
    {
        const TensorShape output_shape = misc::shape_calculator::compute_depthwise_convolution_shape(*input, *weights, conv_info, depth_multiplier, dilation);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

bool NEDepthwiseConvolutionAssemblyDispatch::is_optimized_supported(const ITensorInfo *input,
                                                                    const ITensorInfo *weights,
                                                                    PadStrideInfo      conv_info,
                                                                    unsigned int       depth_multiplier,
                                                                    const Size2D      &dilation)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights);

    // Reshape input shape if in NHWC format
    const DataLayout data_layout = input->data_layout();
    TensorShape      in_shape{ input->tensor_shape() };
    if(data_layout == DataLayout::NHWC)
    {
        in_shape.set(Window::DimX, input->tensor_shape().y());
        in_shape.set(Window::DimY, input->tensor_shape().z());
        in_shape.set(Window::DimZ, input->tensor_shape().x());
    }

    // Check data type
    const DataType data_type          = weights->data_type();
    bool           is_data_type_valid = is_data_type_float(data_type) || is_data_type_quantized_asymmetric(data_type);

    // Check weighs size
    std::set<unsigned int> supported_kernel_sizes = { 3, 5 };
    const unsigned int     width_idx              = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int     height_idx             = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int     kernel_w               = weights->dimension(width_idx);
    const unsigned int     kernel_h               = weights->dimension(height_idx);
    bool                   weights_supported      = (kernel_w == kernel_h) && (supported_kernel_sizes.count(kernel_w) != 0);

    // Check for supported strides
    const auto &strides           = conv_info.stride();
    bool        supported_strides = (strides.first == strides.second) && ((strides.first == 1) || (strides.first == 2));

    // Check for supported padding
    const auto    pad_top           = conv_info.pad_top();
    const auto    pad_right         = conv_info.pad_right();
    const auto    pad_bottom        = conv_info.pad_bottom();
    const auto    pad_left          = conv_info.pad_left();
    PadStrideInfo same_pad          = calculate_same_pad(in_shape, TensorShape(kernel_w, kernel_h), conv_info, DataLayout::NCHW, dilation);
    bool          is_same_padding   = (pad_top == same_pad.pad_top()) && (pad_right == same_pad.pad_right()) && (pad_bottom == same_pad.pad_bottom()) && (pad_left == same_pad.pad_left());
    bool          is_valid_padding  = (pad_top == 0) && (pad_right == 0) && (pad_bottom == 0) && (pad_left == 0);
    bool          supported_padding = is_same_padding || is_valid_padding;
    // TODO(COMPMID-2464): Enable once dilated conv with stride 2 is supported
    bool is_dilation_supported = (dilation == Size2D(1U, 1U)) || ((dilation.x() == dilation.y()) && strides.first == 1);

    return is_data_type_valid && weights_supported && supported_strides && supported_padding && (depth_multiplier == 1) && is_dilation_supported;
}

void NEDepthwiseConvolutionAssemblyDispatch::run()
{
    // Prepare assembly kernel
    prepare();

    MemoryGroupResourceScope scope_mg(_memory_group);

    // Setup inputs/outputs
    ARM_COMPUTE_ERROR_ON(_workspace.buffer() == nullptr);
    _pImpl->_dwc_assembly_kernel->set_working_space(static_cast<void *>(_workspace.buffer()));

    ARM_COMPUTE_ERROR_ON(_input->buffer() == nullptr);
    const int   input_element_size = _input->info()->element_size();
    const int   input_batch_stride = _input->info()->strides_in_bytes()[3] / input_element_size;
    const int   input_row_stride   = _input->info()->strides_in_bytes().z() / input_element_size;
    const int   input_col_stride   = _input->info()->strides_in_bytes().y() / input_element_size;
    const void *input_ptr          = _input->buffer() + _input->info()->offset_first_element_in_bytes();
    _pImpl->_dwc_assembly_kernel->set_input(input_ptr, input_batch_stride, input_row_stride, input_col_stride);

    ARM_COMPUTE_ERROR_ON(_output->buffer() == nullptr);
    const int output_element_size = _output->info()->element_size();
    const int output_batch_stride = _output->info()->strides_in_bytes()[3] / output_element_size;
    const int output_row_stride   = _output->info()->strides_in_bytes().z() / output_element_size;
    const int output_col_stride   = _output->info()->strides_in_bytes().y() / output_element_size;
    void     *output_ptr          = _output->buffer() + _output->info()->offset_first_element_in_bytes();
    _pImpl->_dwc_assembly_kernel->set_output(output_ptr, output_batch_stride, output_row_stride, output_col_stride);

    // Schedule assembly kernel
    NEScheduler::get().schedule(&_pImpl->_dwc_acl_kernel, Window::DimX);
}

void NEDepthwiseConvolutionAssemblyDispatch::prepare()
{
    if(!_is_prepared)
    {
        _packed_weights.allocator()->allocate();
        ARM_COMPUTE_ERROR_ON(_packed_weights.buffer() == nullptr);

        // Pack weights and bias
        const int weights_element_size = _weights->info()->element_size();
        const int weights_row_stride   = _weights->info()->strides_in_bytes().z() / weights_element_size;
        const int weights_col_stride   = _weights->info()->strides_in_bytes().y() / weights_element_size;
        _pImpl->_dwc_assembly_kernel->pack_params(_packed_weights.buffer(),
                                                  _weights->buffer() + _weights->info()->offset_first_element_in_bytes(),
                                                  weights_row_stride,
                                                  weights_col_stride,
                                                  (_bias != nullptr) ? _bias->buffer() : nullptr);
        _pImpl->_dwc_assembly_kernel->set_packed_params_buffer(_packed_weights.buffer());

        _weights->mark_as_unused();
        if(_bias != nullptr)
        {
            _bias->mark_as_unused();
        }
        _is_prepared = true;
    }
}
} // namespace arm_compute
