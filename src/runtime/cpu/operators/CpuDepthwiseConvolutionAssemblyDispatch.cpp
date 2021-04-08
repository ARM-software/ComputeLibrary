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

#include "src/runtime/cpu/operators/CpuDepthwiseConvolutionAssemblyDispatch.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/InfoHelpers.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/kernels/assembly/NEDepthwiseConvolutionAssemblyKernelWrapper.h"
#include "src/core/NEON/kernels/convolution/depthwise/depthwise_dilated.hpp"
#include "src/core/NEON/kernels/convolution/depthwise/depthwise_quantized_dilated.hpp"
#include "src/core/helpers/AutoConfiguration.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

#include <set>

namespace arm_compute
{
namespace cpu
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
                    return std::make_unique<depthwise::QAsymm8DilatedDepthwiseConvolution<2, 2, 3, 3, 1, 1>>(
                               n_batches, in_rows, in_cols, n_channels, dilation_factor, activation, wqinfo, iqinfo, oqinfo, rescale_params, padding_top, padding_left, padding_bottom, padding_right);
                case 2:
                    return std::make_unique<depthwise::QAsymm8DilatedDepthwiseConvolution<2, 2, 3, 3, 2, 2>>(
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
                    return std::make_unique<depthwise::QAsymm8DilatedDepthwiseConvolution<2, 2, 5, 5, 1, 1>>(
                               n_batches, in_rows, in_cols, n_channels, dilation_factor, activation, wqinfo, iqinfo, oqinfo, rescale_params, padding_top, padding_left, padding_bottom, padding_right);
                case 2:
                    return std::make_unique<depthwise::QAsymm8DilatedDepthwiseConvolution<2, 2, 5, 5, 2, 2>>(
                               n_batches, in_rows, in_cols, n_channels, dilation_factor, activation, wqinfo, iqinfo, oqinfo, rescale_params, padding_top, padding_left, padding_bottom, padding_right);
                default:
                    return nullptr;
            }
        }
        default:
            return nullptr;
    }
}

std::unique_ptr<depthwise::IDepthwiseConvolution> get_qsymm8_perchannel_convolver(int kernel_size, int stride_x,
                                                                                  int n_batches, int in_rows, int in_cols, int n_channels,
                                                                                  neon_convolution_kernels::ActivationFunction activation,
                                                                                  const qsymm8::QSymm8PerChannelParams &wqinfo, const qasymm8::QAsymm8Params &iqinfo, const qasymm8::QAsymm8Params &oqinfo,
                                                                                  const qsymm8::QSymm8PerChannelRescaleParams &rescale_params,
                                                                                  int padding_top, int padding_left, int padding_bottom, int padding_right)
{
    switch(kernel_size)
    {
        case 3:
        {
            switch(stride_x)
            {
                case 1:
                    return std::make_unique<depthwise::QSymm8HybridPerChannelDepthwiseConvolution<2, 2, 3, 3, 1, 1>>(
                               n_batches, in_rows, in_cols, n_channels, activation, wqinfo, iqinfo, oqinfo, rescale_params, padding_top, padding_left, padding_bottom, padding_right);
                case 2:
                    return std::make_unique<depthwise::QSymm8HybridPerChannelDepthwiseConvolution<2, 2, 3, 3, 2, 2>>(
                               n_batches, in_rows, in_cols, n_channels, activation, wqinfo, iqinfo, oqinfo, rescale_params, padding_top, padding_left, padding_bottom, padding_right);
                default:
                    return nullptr;
            }
        }
        case 5:
        {
            switch(stride_x)
            {
                case 1:
                    return std::make_unique<depthwise::QSymm8HybridPerChannelDepthwiseConvolution<2, 2, 5, 5, 1, 1>>(
                               n_batches, in_rows, in_cols, n_channels, activation, wqinfo, iqinfo, oqinfo, rescale_params, padding_top, padding_left, padding_bottom, padding_right);
                case 2:
                    return std::make_unique<depthwise::QSymm8HybridPerChannelDepthwiseConvolution<2, 2, 5, 5, 2, 2>>(
                               n_batches, in_rows, in_cols, n_channels, activation, wqinfo, iqinfo, oqinfo, rescale_params, padding_top, padding_left, padding_bottom, padding_right);
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
                    return std::make_unique<depthwise::DilatedDepthwiseConvolution<3, 3, 3, 3, 1, 1, float16_t, float16_t, float16_t>>(
                               n_batches, in_rows, in_cols, n_channels, dilation_factor, activation, padding_top, padding_left, padding_bottom, padding_right);
                case 2:
                    return std::make_unique<depthwise::DilatedDepthwiseConvolution<3, 3, 3, 3, 2, 2, float16_t, float16_t, float16_t>>(
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
                    return std::make_unique<depthwise::DilatedDepthwiseConvolution<3, 3, 5, 5, 1, 1, float16_t, float16_t, float16_t>>(
                               n_batches, in_rows, in_cols, n_channels, dilation_factor, activation, padding_top, padding_left, padding_bottom, padding_right);
                case 2:
                    return std::make_unique<depthwise::DilatedDepthwiseConvolution<3, 3, 5, 5, 2, 2, float16_t, float16_t, float16_t>>(
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
                    return std::make_unique<depthwise::DilatedDepthwiseConvolution<4, 4, 3, 3, 1, 1, float, float, float>>(
                               n_batches, in_rows, in_cols, n_channels, dilation_factor, activation, padding_top, padding_left, padding_bottom, padding_right);
                case 2:
                    return std::make_unique<depthwise::DilatedDepthwiseConvolution<3, 3, 3, 3, 2, 2, float, float, float>>(
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
                    return std::make_unique<depthwise::DilatedDepthwiseConvolution<4, 4, 5, 5, 1, 1, float, float, float>>(
                               n_batches, in_rows, in_cols, n_channels, dilation_factor, activation, padding_top, padding_left, padding_bottom, padding_right);
                case 2:
                    return std::make_unique<depthwise::DilatedDepthwiseConvolution<3, 3, 5, 5, 2, 2, float, float, float>>(
                               n_batches, in_rows, in_cols, n_channels, dilation_factor, activation, padding_top, padding_left, padding_bottom, padding_right);
                default:
                    return nullptr;
            }
        }
        default:
            return nullptr;
    }
}

std::unique_ptr<depthwise::IDepthwiseConvolution> create_convolver(const ITensorInfo     *input,
                                                                   const ITensorInfo     *weights,
                                                                   ITensorInfo           *output,
                                                                   const ConvolutionInfo &info)
{
    const DataType    data_type = input->data_type();
    const TensorShape shape     = input->tensor_shape();

    const int n_batches       = shape[3];
    const int in_rows         = shape.z();
    const int in_cols         = shape.y();
    const int n_channels      = shape.x();
    const int dilation_factor = info.dilation.x();
    const int padding_top     = info.pad_stride_info.pad_top();
    const int padding_left    = info.pad_stride_info.pad_left();
    const int padding_bottom  = info.pad_stride_info.pad_bottom();
    const int padding_right   = info.pad_stride_info.pad_right();

    const bool is_uniform_quantized    = (data_type == DataType::QASYMM8) && (weights->data_type() == DataType::QASYMM8);
    const bool is_perchannel_quantized = (data_type == DataType::QASYMM8) && (weights->data_type() == DataType::QSYMM8_PER_CHANNEL);

    const unsigned int stride_x    = info.pad_stride_info.stride().first;
    const unsigned int kernel_size = weights->tensor_shape().y();

    // Map activation function
    neon_convolution_kernels::ActivationFunction activation = neon_convolution_kernels::ActivationFunction::None;
    if(arm_compute::utils::info_helpers::is_relu(info.act_info))
    {
        activation = neon_convolution_kernels::ActivationFunction::ReLU;
    }
    else if(arm_compute::utils::info_helpers::is_relu6(info.act_info))
    {
        activation = neon_convolution_kernels::ActivationFunction::ReLU6;
    }

    // Create quantized convolver
    if(is_uniform_quantized)
    {
        const UniformQuantizationInfo input_qinfo   = input->quantization_info().uniform();
        const UniformQuantizationInfo weights_qinfo = weights->quantization_info().uniform();
        const UniformQuantizationInfo output_qinfo  = output->quantization_info().uniform();

        // Check that quantization info are in the range [0, 255]
        ARM_COMPUTE_ERROR_ON(input_qinfo.offset < 0 || input_qinfo.offset > 255);
        ARM_COMPUTE_ERROR_ON(weights_qinfo.offset < 0 || weights_qinfo.offset > 255);
        ARM_COMPUTE_ERROR_ON(output_qinfo.offset < 0 || output_qinfo.offset > 255);
        const qasymm8::QAsymm8Params iqinfo{ static_cast<uint8_t>(input_qinfo.offset), input_qinfo.scale };
        const qasymm8::QAsymm8Params wqinfo{ static_cast<uint8_t>(weights_qinfo.offset), weights_qinfo.scale };
        const qasymm8::QAsymm8Params oqinfo{ static_cast<uint8_t>(output_qinfo.offset), output_qinfo.scale };

        // Calculate rescale parameters
        const float fmultipler  = iqinfo.scale * wqinfo.scale / oqinfo.scale;
        int32_t     qmultiplier = 0;
        int32_t     qshift      = 0;
        quantization::calculate_quantized_multiplier_less_than_one(fmultipler, &qmultiplier, &qshift);
        qasymm8::QAsymm8RescaleParams rescale_params(qshift, qmultiplier, fmultipler);

        return get_qasymm8_convolver(kernel_size, stride_x, n_batches, in_rows, in_cols, n_channels, dilation_factor, activation,
                                     wqinfo, iqinfo, oqinfo, rescale_params, padding_top, padding_left, padding_bottom, padding_right);
    }
    else if(is_perchannel_quantized)
    {
        const UniformQuantizationInfo input_qinfo   = input->quantization_info().uniform();
        const QuantizationInfo        weights_qinfo = weights->quantization_info();
        const UniformQuantizationInfo output_qinfo  = output->quantization_info().uniform();

        // Check that quantization info are in the range [0, 255]
        ARM_COMPUTE_ERROR_ON(input_qinfo.offset < 0 || input_qinfo.offset > 255);
        ARM_COMPUTE_ERROR_ON(output_qinfo.offset < 0 || output_qinfo.offset > 255);
        const qasymm8::QAsymm8Params         iqinfo{ static_cast<uint8_t>(input_qinfo.offset), input_qinfo.scale };
        const qsymm8::QSymm8PerChannelParams wqinfo{ weights_qinfo.scale() };
        const qasymm8::QAsymm8Params         oqinfo{ static_cast<uint8_t>(output_qinfo.offset), output_qinfo.scale };

        // Calculate rescale parameters
        std::vector<float>   fmultipliers;
        std::vector<int32_t> qmultipliers;
        std::vector<int32_t> qshifts;

        for(auto const s : wqinfo.scales)
        {
            const float fmultipler  = iqinfo.scale * s / oqinfo.scale;
            int32_t     qmultiplier = 0;
            int32_t     qshift      = 0;
            quantization::calculate_quantized_multiplier_less_than_one(fmultipler, &qmultiplier, &qshift);
            fmultipliers.push_back(fmultipler);
            qmultipliers.push_back(qmultiplier);
            qshifts.push_back(qshift);
        }

        qsymm8::QSymm8PerChannelRescaleParams rescale_params(qshifts, qmultipliers, fmultipliers);

        return get_qsymm8_perchannel_convolver(kernel_size, stride_x, n_batches, in_rows, in_cols, n_channels, activation,
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

struct CpuDepthwiseConvolutionAssemblyDispatch::LocalImpl
{
    std::unique_ptr<depthwise::IDepthwiseConvolution> dwc_assembly_kernel{ nullptr };
    NEDepthwiseConvolutionAssemblyKernelWrapper       dwc_acl_kernel{};
    bool                                              is_prepared{ false };
    experimental::MemoryRequirements                  mem_req{};
};

#ifndef DOXYGEN_SKIP_THIS
CpuDepthwiseConvolutionAssemblyDispatch::CpuDepthwiseConvolutionAssemblyDispatch()
    : _pImpl(std::make_unique<LocalImpl>())
{
}
#endif /* DOXYGEN_SKIP_THIS */

CpuDepthwiseConvolutionAssemblyDispatch::~CpuDepthwiseConvolutionAssemblyDispatch() = default;

void CpuDepthwiseConvolutionAssemblyDispatch::configure(const ITensorInfo     *input,
                                                        const ITensorInfo     *weights,
                                                        const ITensorInfo     *bias,
                                                        ITensorInfo           *output,
                                                        const ConvolutionInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_UNUSED(bias);
    ARM_COMPUTE_ERROR_THROW_ON(CpuDepthwiseConvolutionAssemblyDispatch::validate(input,
                                                                                 weights,
                                                                                 bias != nullptr ? bias : nullptr,
                                                                                 output,
                                                                                 info));

    // Output auto inizialitation if not yet initialized
    const TensorShape output_shape = misc::shape_calculator::compute_depthwise_convolution_shape(*input, *weights, info);
    auto_init_if_empty(*output, input->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(output_shape).set_quantization_info(output->quantization_info()));

    _pImpl->is_prepared = false;

    // Create convolver
    _pImpl->dwc_assembly_kernel = create_convolver(input, weights, output, info);
    ARM_COMPUTE_ERROR_ON(_pImpl->dwc_assembly_kernel == nullptr);

    // Create assembly kernel wrapper
    _pImpl->dwc_acl_kernel.configure(_pImpl->dwc_assembly_kernel.get());

    constexpr size_t alignment = 128;

    // Create workspace
    const unsigned int num_threads    = NEScheduler::get().num_threads();
    const size_t       workspace_size = _pImpl->dwc_assembly_kernel->get_working_space_size(num_threads);
    ARM_COMPUTE_ERROR_ON_MSG(workspace_size == 0, "Workspace size cannot be 0 !");
    _pImpl->mem_req.push_back({ TensorType::ACL_INT_0, workspace_size, alignment });

    // Create packing tensor
    const size_t pack_tensor_size = _pImpl->dwc_assembly_kernel->get_packed_params_size();
    ARM_COMPUTE_ERROR_ON_MSG(pack_tensor_size == 0, "Pack tensor size cannot be 0 !");

    _pImpl->mem_req.push_back({ TensorType::ACL_INT_1, pack_tensor_size, alignment });
}

experimental::MemoryRequirements CpuDepthwiseConvolutionAssemblyDispatch::workspace() const
{
    return _pImpl->mem_req;
}

Status CpuDepthwiseConvolutionAssemblyDispatch::validate(const ITensorInfo     *input,
                                                         const ITensorInfo     *weights,
                                                         const ITensorInfo     *bias,
                                                         const ITensorInfo     *output,
                                                         const ConvolutionInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    if(weights->data_type() != DataType::QSYMM8_PER_CHANNEL)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    }
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, weights);

    // Validate convolver
    ARM_COMPUTE_RETURN_ERROR_ON(!is_optimized_supported(input, weights, info));

    // Validate activation
    const bool is_relu  = arm_compute::utils::info_helpers::is_relu(info.act_info);
    const bool is_relu6 = arm_compute::utils::info_helpers::is_relu6(info.act_info);
    ARM_COMPUTE_RETURN_ERROR_ON(info.act_info.enabled() && !(is_relu || is_relu6));

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
        const TensorShape output_shape = misc::shape_calculator::compute_depthwise_convolution_shape(*input, *weights, info);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    // The uniform quantization case will only have 1 scale value in the weights quantization info
    const UniformQuantizationInfo input_qinfo   = input->quantization_info().uniform();
    const QuantizationInfo        weights_qinfo = weights->quantization_info();
    const UniformQuantizationInfo output_qinfo  = output->quantization_info().uniform();
    for(auto const s : weights_qinfo.scale())
    {
        const float fmultipler = input_qinfo.scale * s / output_qinfo.scale;
        ARM_COMPUTE_RETURN_ERROR_ON(fmultipler > 1.f);
    }

    return Status{};
}

bool CpuDepthwiseConvolutionAssemblyDispatch::is_optimized_supported(const ITensorInfo     *input,
                                                                     const ITensorInfo     *weights,
                                                                     const ConvolutionInfo &info)
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
    // TODO (COMPMID-3004): Add assembly optimized routine for QASYMM8_SIGNED NEDepthwiseConvolutionLayer
    const DataType input_type            = input->data_type();
    const bool     is_input_type_valid   = is_data_type_float(input_type) || input_type == DataType::QASYMM8;
    const DataType weights_type          = weights->data_type();
    const bool     is_weights_type_valid = is_data_type_float(weights_type) || weights_type == DataType::QASYMM8 || weights_type == DataType::QASYMM8_SIGNED
                                           || weights_type == DataType::QSYMM8_PER_CHANNEL;

    // Check weighs size
    std::set<unsigned int> supported_kernel_sizes = { 3, 5 };
    const unsigned int     width_idx              = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int     height_idx             = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int     kernel_w               = weights->dimension(width_idx);
    const unsigned int     kernel_h               = weights->dimension(height_idx);
    bool                   weights_supported      = (kernel_w == kernel_h) && (supported_kernel_sizes.count(kernel_w) != 0);

    // Check for supported strides
    const auto &strides           = info.pad_stride_info.stride();
    bool        supported_strides = (strides.first == strides.second) && ((strides.first == 1) || (strides.first == 2));

    // Check for supported padding
    const auto    pad_top           = info.pad_stride_info.pad_top();
    const auto    pad_right         = info.pad_stride_info.pad_right();
    const auto    pad_bottom        = info.pad_stride_info.pad_bottom();
    const auto    pad_left          = info.pad_stride_info.pad_left();
    PadStrideInfo same_pad          = calculate_same_pad(in_shape, TensorShape(kernel_w, kernel_h), info.pad_stride_info, DataLayout::NCHW, info.dilation);
    bool          is_same_padding   = (pad_top == same_pad.pad_top()) && (pad_right == same_pad.pad_right()) && (pad_bottom == same_pad.pad_bottom()) && (pad_left == same_pad.pad_left());
    bool          is_valid_padding  = (pad_top == 0) && (pad_right == 0) && (pad_bottom == 0) && (pad_left == 0);
    bool          supported_padding = is_same_padding || is_valid_padding;
    // TODO(COMPMID-2464): Enable once dilated conv with stride 2 is supported
    bool is_dilation_supported = ((info.dilation == Size2D(1U, 1U)) || ((info.dilation.x() == info.dilation.y()) && strides.first == 1));

    if(weights_type == DataType::QSYMM8_PER_CHANNEL)
    {
        is_dilation_supported = is_dilation_supported && (info.dilation == Size2D(1U, 1U));
    }

    return is_input_type_valid && is_weights_type_valid && weights_supported && supported_strides && supported_padding && (info.depth_multiplier == 1) && is_dilation_supported;
}

void CpuDepthwiseConvolutionAssemblyDispatch::run(ITensorPack &tensors)
{
    // Prepare assembly kernel
    prepare(tensors);

    auto src       = tensors.get_tensor(TensorType::ACL_SRC_0);
    auto workspace = tensors.get_tensor(TensorType::ACL_INT_0);
    auto dst       = tensors.get_tensor(TensorType::ACL_DST);

    // Setup inputs/outputs
    ARM_COMPUTE_ERROR_ON(workspace == nullptr && workspace->buffer() == nullptr);
    _pImpl->dwc_assembly_kernel->set_working_space(static_cast<void *>(workspace->buffer()));

    ARM_COMPUTE_ERROR_ON(workspace->buffer() == nullptr);
    const int   input_element_size = src->info()->element_size();
    const int   input_batch_stride = src->info()->strides_in_bytes()[3] / input_element_size;
    const int   input_row_stride   = src->info()->strides_in_bytes().z() / input_element_size;
    const int   input_col_stride   = src->info()->strides_in_bytes().y() / input_element_size;
    const void *input_ptr          = src->buffer() + src->info()->offset_first_element_in_bytes();
    _pImpl->dwc_assembly_kernel->set_input(input_ptr, input_batch_stride, input_row_stride, input_col_stride);

    ARM_COMPUTE_ERROR_ON(dst->buffer() == nullptr);
    const int output_element_size = dst->info()->element_size();
    const int output_batch_stride = dst->info()->strides_in_bytes()[3] / output_element_size;
    const int output_row_stride   = dst->info()->strides_in_bytes().z() / output_element_size;
    const int output_col_stride   = dst->info()->strides_in_bytes().y() / output_element_size;
    void     *output_ptr          = dst->buffer() + dst->info()->offset_first_element_in_bytes();
    _pImpl->dwc_assembly_kernel->set_output(output_ptr, output_batch_stride, output_row_stride, output_col_stride);

    // Schedule assembly kernel
    NEScheduler::get().schedule(&_pImpl->dwc_acl_kernel, Window::DimX);
}

void CpuDepthwiseConvolutionAssemblyDispatch::prepare(ITensorPack &tensors)
{
    if(!_pImpl->is_prepared)
    {
        auto weights        = tensors.get_const_tensor(TensorType::ACL_SRC_1);
        auto bias           = tensors.get_const_tensor(TensorType::ACL_SRC_2);
        auto packed_weights = tensors.get_tensor(TensorType::ACL_INT_1);

        ARM_COMPUTE_ERROR_ON(packed_weights->buffer() == nullptr);

        // Pack weights and bias
        const int weights_element_size = weights->info()->element_size();
        const int weights_row_stride   = weights->info()->strides_in_bytes().z() / weights_element_size;
        const int weights_col_stride   = weights->info()->strides_in_bytes().y() / weights_element_size;
        _pImpl->dwc_assembly_kernel->pack_params(packed_weights->buffer(),
                                                 weights->buffer() + weights->info()->offset_first_element_in_bytes(),
                                                 weights_row_stride,
                                                 weights_col_stride,
                                                 (bias != nullptr) ? bias->buffer() : nullptr);
        _pImpl->dwc_assembly_kernel->set_packed_params_buffer(packed_weights->buffer());

        weights->mark_as_unused();
        if(bias != nullptr)
        {
            bias->mark_as_unused();
        }
        _pImpl->is_prepared = true;
    }
}
} // namespace cpu
} // namespace arm_compute
