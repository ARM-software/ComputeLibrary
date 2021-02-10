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
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/NEON/functions/NEDirectConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFFTConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMConv2d.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEWinogradConvolutionLayer.h"

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
                                                            enable_fast_math, num_groups));

    const Conv2dInfo info(conv_info, dilation, act_info, enable_fast_math, num_groups);
    switch(NEConvolutionLayer::get_convolution_method(input->info(), weights->info(), output->info(), conv_info, weights_info, dilation, act_info, enable_fast_math))
    {
        case ConvolutionMethod::WINOGRAD:
        {
            auto f = std::make_unique<NEWinogradConvolutionLayer>(_memory_manager);
            f->configure(input, weights, biases, output, conv_info, act_info, enable_fast_math);
            _function = std::move(f);
            break;
        }
        case ConvolutionMethod::GEMM:
        {
            auto f = std::make_unique<NEGEMMConvolutionLayer>(_memory_manager);
            f->configure(input, weights, biases, output, conv_info, weights_info, dilation, act_info);
            _function = std::move(f);
            break;
        }
        case ConvolutionMethod::GEMM_CONV2D:
        {
            auto f = std::make_unique<NEGEMMConv2d>(_memory_manager);
            f->configure(input, weights, biases, output, info);
            _function = std::move(f);
            break;
        }
        case ConvolutionMethod::DIRECT:
        {
            auto f = std::make_unique<NEDirectConvolutionLayer>(_memory_manager);
            f->configure(input, weights, biases, output, conv_info, act_info);
            _function = std::move(f);
            break;
        }
        case ConvolutionMethod::FFT:
        {
            auto f = std::make_unique<NEFFTConvolutionLayer>(_memory_manager);
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
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((num_groups != 1), "Grouping (num_groups != 1) is not supported on Neon");

    const Conv2dInfo info(conv_info, dilation, act_info, enable_fast_math, num_groups);
    switch(NEConvolutionLayer::get_convolution_method(input, weights, output, conv_info, weights_info, dilation, act_info, enable_fast_math))
    {
        case ConvolutionMethod::WINOGRAD:
            ARM_COMPUTE_RETURN_ON_ERROR(NEWinogradConvolutionLayer::validate(input, weights, biases, output, conv_info, act_info, enable_fast_math));
            break;
        case ConvolutionMethod::GEMM:
            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMConvolutionLayer::validate(input, weights, biases, output, conv_info, weights_info, dilation, act_info));
            break;
        case ConvolutionMethod::GEMM_CONV2D:
            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMConv2d::validate(input, weights, biases, output, info));
            break;
        case ConvolutionMethod::DIRECT:
            ARM_COMPUTE_RETURN_ON_ERROR(NEDirectConvolutionLayer::validate(input, weights, biases, output, conv_info, act_info));
            break;
        case ConvolutionMethod::FFT:
            ARM_COMPUTE_RETURN_ON_ERROR(NEFFTConvolutionLayer::validate(input, weights, nullptr, output, conv_info, act_info));
            break;
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

    const Conv2dInfo info(conv_info, dilation, act_info, enable_fast_math, 1);

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

    if(dilation != Size2D(1U, 1U))
    {
        return ConvolutionMethod::GEMM;
    }
    else
    {
        // SRGAN
        // Output might not be initialized when it is an internal tensor of the layer using the convolution
        if(input->total_size() > 1e7 && (weights->dimension(idx_h) > 7)
           && (NEDirectConvolutionLayer::validate(input, weights, nullptr, output, conv_info, act_info)))
        {
            return ConvolutionMethod::DIRECT;
        }
        if((weights->dimension(idx_h) > 7) && (input->dimension(idx_c) > output->dimension(idx_c)) && (NEFFTConvolutionLayer::validate(input, weights, nullptr, output, conv_info, act_info)))
        {
            return ConvolutionMethod::FFT;
        }
        if(input->dimension(idx_c) < 16)
        {
            return ConvolutionMethod::GEMM;
        }

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        // This heuristics only applies to F16 data type on A55r1
        if(NEScheduler::get().cpu_info().get_cpu_model() == CPUModel::A55r1 && enable_fast_math && input->data_type() == DataType::F16)
        {
            // Exclude known bad winograd configs (and defaults to GEMM)
            const std::vector<ConvolutionConfiguration> known_bad_winograd_f16_with_fastmath_configs =
            {
                // Squeezenet_V1_1 fire2 and fire3
                ConvolutionConfiguration(Size2D(56U, 56U), Size2D(3U, 3U), Size2D(16U, 64U), PadStrideInfo(1U, 1U, 1U, 1U)),
                // Squeezenet_V1_1 fire6 and fire7
                ConvolutionConfiguration(Size2D(14U, 14U), Size2D(3U, 3U), Size2D(48U, 192U), PadStrideInfo(1U, 1U, 1U, 1U)),
                // Squeezenet_V1_1 fire8 and fire9
                ConvolutionConfiguration(Size2D(14U, 14U), Size2D(3U, 3U), Size2D(64U, 256U), PadStrideInfo(1U, 1U, 1U, 1U)),
            };
            const auto find_conv_config = [&](ConvolutionConfiguration c)
            {
                const PadStrideInfo info = std::get<3>(c);

                return std::get<0>(c) == Size2D(input->dimension(idx_w), input->dimension(idx_h)) && std::get<1>(c) == Size2D(weights->dimension(idx_w), weights->dimension(idx_h))
                       && std::get<2>(c) == Size2D(weights->dimension(idx_c), weights->dimension(3)) && info.pad_top() == conv_info.pad_top() && info.pad_right() == conv_info.pad_right()
                       && info.pad_bottom() == conv_info.pad_bottom() && info.pad_left() == conv_info.pad_left() && info.stride() == conv_info.stride();
            };

            bool found_bad = std::find_if(known_bad_winograd_f16_with_fastmath_configs.begin(), known_bad_winograd_f16_with_fastmath_configs.end(),
                                          find_conv_config)
                             != known_bad_winograd_f16_with_fastmath_configs.end();
            if(found_bad)
            {
                return ConvolutionMethod::GEMM;
            }
        }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        // For 1x1 convolutions run the default GEMM
        if(weights->dimension(idx_w) == 1 && weights->dimension(idx_h) == 1)
        {
            return ConvolutionMethod::GEMM;
        }

        if(bool(NEWinogradConvolutionLayer::validate(input, weights, nullptr, output, conv_info, act_info, enable_fast_math)))
        {
            return ConvolutionMethod::WINOGRAD;
        }
        if(bool(NEGEMMConv2d::validate(input, weights, nullptr, output, info)))
        {
            return ConvolutionMethod::GEMM_CONV2D;
        }
        return ConvolutionMethod::GEMM;
    }
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
