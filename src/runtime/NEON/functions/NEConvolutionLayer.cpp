/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "support/ToolchainSupport.h"

#include <cmath>
#include <tuple>
#include <utility>

namespace arm_compute
{
NEConvolutionLayer::NEConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager) //NOLINT
    : _memory_manager(std::move(memory_manager)),
      _function()
{
}

void NEConvolutionLayer::configure(ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info,
                                   const Size2D &dilation, const ActivationLayerInfo &act_info, bool enable_fast_math, unsigned int num_groups)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_UNUSED(num_groups);
    ARM_COMPUTE_ERROR_THROW_ON(NEConvolutionLayer::validate(input->info(), weights->info(), ((biases != nullptr) ? biases->info() : nullptr), output->info(), conv_info, weights_info, dilation, act_info,
                                                            enable_fast_math));

    switch(NEConvolutionLayer::get_convolution_method(input->info(), weights->info(), output->info(), conv_info, weights_info, dilation, act_info))
    {
        case ConvolutionMethod::WINOGRAD:
        {
            auto f = arm_compute::support::cpp14::make_unique<NEWinogradConvolutionLayer>(_memory_manager);
            f->configure(input, weights, biases, output, conv_info, act_info, enable_fast_math);
            _function = std::move(f);
            break;
        }
        case ConvolutionMethod::GEMM:
        {
            auto f = arm_compute::support::cpp14::make_unique<NEGEMMConvolutionLayer>(_memory_manager);
            f->configure(input, weights, biases, output, conv_info, weights_info, dilation, act_info);
            _function = std::move(f);
            break;
        }
        case ConvolutionMethod::DIRECT:
        {
            auto f = arm_compute::support::cpp14::make_unique<NEDirectConvolutionLayer>(_memory_manager);
            f->configure(input, weights, biases, output, conv_info, act_info);
            _function = std::move(f);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported.");
            break;
    }
}

Status NEConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                    const WeightsInfo &weights_info, const Size2D &dilation, const ActivationLayerInfo &act_info, bool enable_fast_math, unsigned int num_groups)
{
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((num_groups != 1), "Grouping (num_groups != 1) is not supported on NEON");

    switch(NEConvolutionLayer::get_convolution_method(input, weights, output, conv_info, weights_info, dilation, act_info))
    {
        case ConvolutionMethod::WINOGRAD:
            //Validate Winograd
            ARM_COMPUTE_RETURN_ON_ERROR(NEWinogradConvolutionLayer::validate(input, weights, biases, output, conv_info, act_info, enable_fast_math));
            break;
        case ConvolutionMethod::GEMM:
            //Validate Gemm-based Convolution
            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMConvolutionLayer::validate(input, weights, biases, output, conv_info, weights_info, dilation, act_info));
            break;
        case ConvolutionMethod::DIRECT:
            //Validate Gemm-based Convolution
            ARM_COMPUTE_RETURN_ON_ERROR(NEDirectConvolutionLayer::validate(input, weights, biases, output, conv_info, act_info));
        default:
            ARM_COMPUTE_ERROR("Not supported.");
            break;
    }

    return Status{};
}

ConvolutionMethod NEConvolutionLayer::get_convolution_method(const ITensorInfo *input, const ITensorInfo *weights,
                                                             const ITensorInfo *output, const PadStrideInfo &conv_info,
                                                             const WeightsInfo &weights_info, const Size2D &dilation, const ActivationLayerInfo &act_info, bool enable_fast_math)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, weights);
    ARM_COMPUTE_UNUSED(weights_info);

    const size_t idx_w = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);
    const size_t idx_c = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL);

    /* Input spatial dims, kernel size, IFM/OFM, conv info*/
    using ConvolutionConfiguration = std::tuple<Size2D, Size2D, Size2D, PadStrideInfo>;
    using ConfigurationMethod      = std::pair<ConvolutionConfiguration, ConvolutionMethod>;

    const std::vector<ConfigurationMethod> known_configs =
    {
        // Alexnet
        ConfigurationMethod(ConvolutionConfiguration(Size2D(27U, 27U), Size2D(5U, 5U), Size2D(48U, 128U), PadStrideInfo(1U, 1U, 2U, 2U)), ConvolutionMethod::GEMM),
        // VGG16 / VGG19
        ConfigurationMethod(ConvolutionConfiguration(Size2D(224U, 224U), Size2D(3U, 3U), Size2D(3U, 64U), PadStrideInfo(1U, 1U, 1U, 1U)), ConvolutionMethod::GEMM),
        // Mobilenet 224
        ConfigurationMethod(ConvolutionConfiguration(Size2D(224U, 224U), Size2D(3U, 3U), Size2D(3U, 32U), PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR)), ConvolutionMethod::GEMM),
        // Mobilenet 160
        ConfigurationMethod(ConvolutionConfiguration(Size2D(160U, 160U), Size2D(3U, 3U), Size2D(3U, 24U), PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR)), ConvolutionMethod::GEMM)
    };

    const auto find_config = [&](ConfigurationMethod c)
    {
        const ConvolutionConfiguration config = c.first;
        const PadStrideInfo            info   = std::get<3>(config);

        return std::get<0>(config) == Size2D(input->dimension(idx_w), input->dimension(idx_h)) && std::get<1>(config) == Size2D(weights->dimension(idx_w), weights->dimension(idx_h))
               && std::get<2>(config) == Size2D(weights->dimension(idx_c), weights->dimension(3)) && info.pad_top() == conv_info.pad_top() && info.pad_right() == conv_info.pad_right()
               && info.pad_bottom() == conv_info.pad_bottom() && info.pad_left() == conv_info.pad_left() && info.stride() == conv_info.stride();
    };

    std::vector<ConfigurationMethod>::const_iterator found;
    if((found = std::find_if(known_configs.begin(), known_configs.end(), find_config)) != known_configs.end())
    {
        return (*found).second;
    }

    if(dilation != Size2D(1U, 1U) || input->dimension(get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL)) <= 16)
    {
        return ConvolutionMethod::GEMM;
    }

    return bool(NEWinogradConvolutionLayer::validate(input, weights, nullptr, output, conv_info, act_info, enable_fast_math)) ? ConvolutionMethod::WINOGRAD : ConvolutionMethod::GEMM;
}

void NEConvolutionLayer::run()
{
    prepare();
    _function->run();
}

void NEConvolutionLayer::prepare()
{
    _function->prepare();
}
} // namespace arm_compute
