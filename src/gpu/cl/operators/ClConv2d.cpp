/*
 * Copyright (c) 2021-2022 Arm Limited.
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
#include "src/gpu/cl/operators/ClConv2d.h"

#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/functions/CLFFTConvolutionLayer.h"
#include "src/gpu/cl/operators/ClDirectConv2d.h"
#include "src/gpu/cl/operators/ClGemmConv2d.h"
#include "src/gpu/cl/operators/ClIndirectConv2d.h"
#include "src/gpu/cl/operators/ClWinogradConv2d.h"

#include "src/common/utils/Log.h"

#include <memory>

namespace
{
/** Get the suitable kernel size for using direct convolution method with NHWC data layout.
 *
 * @note Direct convolution should be executed when the kernel has the spatial dimensions greater than or equal to the value returned by this function
 *
 * @param[in] gpu_target GPU target
 *
 * @return the suitable kernel size for using direct convolution method with NHWC data layout
 */
size_t get_direct_conv_kernel_threshold_nhwc(arm_compute::GPUTarget gpu_target)
{
    switch(gpu_target)
    {
        case arm_compute::GPUTarget::G76:
        case arm_compute::GPUTarget::G77:
        case arm_compute::GPUTarget::G78:
            return 5;
        case arm_compute::GPUTarget::G71:
        case arm_compute::GPUTarget::G72:
        case arm_compute::GPUTarget::MIDGARD:
        case arm_compute::GPUTarget::BIFROST:
            return 7;
        default:
            return 5;
    }
}
} // namespace

namespace arm_compute
{
namespace opencl
{
using namespace arm_compute::misc::shape_calculator;

ClConv2d::ClConv2d()
    : _operator()
{
}

ClConv2d::~ClConv2d() = default;

void ClConv2d::configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *weights, ITensorInfo *biases, ITensorInfo *dst, const Conv2dInfo &conv2d_info,
                         const WeightsInfo &weights_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_ERROR_THROW_ON(ClConv2d::validate(src, weights, ((biases != nullptr) ? biases : nullptr), dst, conv2d_info, weights_info));
    ARM_COMPUTE_LOG_PARAMS(src, weights, biases, dst, conv2d_info, weights_info);

    switch(ClConv2d::get_convolution_method(src, weights, dst, conv2d_info, weights_info, CLScheduler::get().target()))
    {
        case ConvolutionMethod::WINOGRAD:
        {
            ARM_COMPUTE_ERROR_ON(conv2d_info.num_groups != 1);
            ARM_COMPUTE_ERROR_ON(conv2d_info.post_ops.size() > 0);
            auto f = std::make_unique<ClWinogradConv2d>();
            f->configure(compile_context, src, weights, biases, dst, conv2d_info.conv_info, conv2d_info.act_info, conv2d_info.enable_fast_math);
            _operator = std::move(f);
            break;
        }
        case ConvolutionMethod::DIRECT:
        {
            ARM_COMPUTE_ERROR_ON(conv2d_info.num_groups != 1);
            ARM_COMPUTE_ERROR_ON(conv2d_info.post_ops.size() > 0);
            auto f = std::make_unique<ClDirectConv2d>();
            f->configure(compile_context, src, weights, biases, dst, conv2d_info.conv_info, conv2d_info.act_info);
            _operator = std::move(f);
            break;
        }
        case ConvolutionMethod::INDIRECT:
        {
            ARM_COMPUTE_ERROR_ON(conv2d_info.num_groups != 1);
            ARM_COMPUTE_ERROR_ON(conv2d_info.post_ops.size() > 0);
            auto f = std::make_unique<ClIndirectConv2d>();
            f->configure(compile_context, src, weights, biases, dst, conv2d_info.conv_info, conv2d_info.act_info);
            _operator = std::move(f);
            break;
        }
        case ConvolutionMethod::GEMM:
        {
            auto f = std::make_unique<ClGemmConv2d>();
            f->configure(compile_context, src, weights, biases, dst, conv2d_info, weights_info);
            _operator = std::move(f);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported.");
            break;
    }
    _aux_mem = _operator->workspace();
}

Status ClConv2d::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const Conv2dInfo &conv2d_info,
                          const WeightsInfo &weights_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((conv2d_info.num_groups != 1) && (src->data_layout() != DataLayout::NCHW), "Grouping (num_groups != 1) with NHWC data layout is not supported");

    const GPUTarget gpu_target = CLScheduler::get().target();

    switch(ClConv2d::get_convolution_method(src, weights, dst, conv2d_info, weights_info, gpu_target))
    {
        case ConvolutionMethod::WINOGRAD:
        {
            //Validate Winograd
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv2d_info.num_groups != 1, "Grouping (num_groups != 1) with ClWinogradConv2d is not supported");
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv2d_info.post_ops.size() > 0, "ClWinogradConv2d does not support PostOps");
            ARM_COMPUTE_RETURN_ON_ERROR(ClWinogradConv2d::validate(src, weights, biases, dst, conv2d_info.conv_info, conv2d_info.act_info, conv2d_info.enable_fast_math));
            break;
        }
        case ConvolutionMethod::DIRECT:
        {
            // Validate direct convolution layer
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv2d_info.num_groups != 1, "Grouping (num_groups != 1) with ClDirectConv2d is not supported");
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv2d_info.post_ops.size() > 0, "ClDirectConv2d does not support PostOps");
            ARM_COMPUTE_RETURN_ON_ERROR(ClDirectConv2d::validate(src, weights, biases, dst, conv2d_info.conv_info, conv2d_info.act_info));
            break;
        }
        case ConvolutionMethod::INDIRECT:
        {
            // Validate indirect convolution layer
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv2d_info.num_groups != 1, "Grouping (num_groups != 1) with ClIndirectConv2d is not supported");
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv2d_info.post_ops.size() > 0, "ClIndirectConv2d does not support PostOps");
            ARM_COMPUTE_RETURN_ON_ERROR(ClIndirectConv2d::validate(src, weights, biases, dst, conv2d_info.conv_info, conv2d_info.act_info));
            break;
        }
        case ConvolutionMethod::GEMM:
        {
            // Validate gemm-based convolution layer
            ARM_COMPUTE_RETURN_ON_ERROR(ClGemmConv2d::validate(src, weights, biases, dst, conv2d_info, weights_info));
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Not supported.");
            break;
    }

    return Status{};
}

ConvolutionMethod ClConv2d::get_convolution_method(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *dst, const Conv2dInfo &conv2d_info,
                                                   const WeightsInfo &weights_info, const GPUTarget gpu_target)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src);
    ARM_COMPUTE_ERROR_ON_NULLPTR(dst);
    ARM_COMPUTE_ERROR_ON_NULLPTR(weights);
    ARM_COMPUTE_UNUSED(weights_info);

    const PadStrideInfo       conv_info        = conv2d_info.conv_info;
    const ActivationLayerInfo act_info         = conv2d_info.act_info;
    const Size2D              dilation         = conv2d_info.dilation;
    bool                      enable_fast_math = conv2d_info.enable_fast_math;

    const size_t idx_w = get_data_layout_dimension_index(src->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(src->data_layout(), DataLayoutDimension::HEIGHT);
    const size_t idx_c = get_data_layout_dimension_index(src->data_layout(), DataLayoutDimension::CHANNEL);

    /* Input spatial dims, kernel size, IFM/OFM, conv info*/
    using ConvolutionConfiguration = std::tuple<Size2D, Size2D, Size2D, PadStrideInfo, DataLayout>;
    using ConfigurationMethod      = std::pair<ConvolutionConfiguration, ConvolutionMethod>;

    const std::vector<ConfigurationMethod> known_configs =
    {
        // Alexnet
        ConfigurationMethod(ConvolutionConfiguration(Size2D(27U, 27U), Size2D(5U, 5U), Size2D(48U, 128U), PadStrideInfo(1U, 1U, 2U, 2U), DataLayout::NCHW), ConvolutionMethod::DIRECT),
        // VGG16 / VGG19
        ConfigurationMethod(ConvolutionConfiguration(Size2D(224U, 224U), Size2D(3U, 3U), Size2D(3U, 64U), PadStrideInfo(1U, 1U, 1U, 1U), DataLayout::NCHW), ConvolutionMethod::DIRECT),
        // Mobilenet 224
        ConfigurationMethod(ConvolutionConfiguration(Size2D(224U, 224U), Size2D(3U, 3U), Size2D(3U, 32U), PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), DataLayout::NCHW), ConvolutionMethod::GEMM),
        // Mobilenet 160
        ConfigurationMethod(ConvolutionConfiguration(Size2D(160U, 160U), Size2D(3U, 3U), Size2D(3U, 24U), PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), DataLayout::NCHW), ConvolutionMethod::GEMM),
        // Mobilenet 224
        ConfigurationMethod(ConvolutionConfiguration(Size2D(224U, 224U), Size2D(3U, 3U), Size2D(3U, 32U), PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), DataLayout::NHWC), ConvolutionMethod::GEMM),
        // Mobilenet 160
        ConfigurationMethod(ConvolutionConfiguration(Size2D(160U, 160U), Size2D(3U, 3U), Size2D(3U, 24U), PadStrideInfo(2U, 2U, 0U, 1U, 0U, 1U, DimensionRoundingType::FLOOR), DataLayout::NHWC), ConvolutionMethod::GEMM),
    };

    const auto find_config = [&](ConfigurationMethod c)
    {
        const ConvolutionConfiguration config      = c.first;
        const PadStrideInfo            info        = std::get<3>(config);
        const DataLayout               data_layout = std::get<4>(config);

        return std::get<0>(config) == Size2D(src->dimension(idx_w), src->dimension(idx_h)) && std::get<1>(config) == Size2D(weights->dimension(idx_w), weights->dimension(idx_h))
               && std::get<2>(config) == Size2D(weights->dimension(idx_c), weights->dimension(3)) && info.pad_top() == conv_info.pad_top() && info.pad_right() == conv_info.pad_right()
               && info.pad_bottom() == conv_info.pad_bottom() && info.pad_left() == conv_info.pad_left() && info.stride() == conv_info.stride() && (data_layout == src->data_layout());
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
        if(src->data_layout() == DataLayout::NCHW)
        {
            // SRGAN
            if((src->dimension(idx_h) > 720U) && (dst->dimension(idx_h) > 720U) && (weights->dimension(idx_h) == 9) && (conv_info.pad_top() < 3)
               && (ClDirectConv2d::validate(src, weights, nullptr, dst, conv_info, act_info)))
            {
                return ConvolutionMethod::DIRECT;
            }
            if((weights->dimension(idx_h) > 5) && (src->dimension(idx_c) > dst->dimension(idx_c)) && (CLFFTConvolutionLayer::validate(src, weights, nullptr, dst, conv_info, act_info, enable_fast_math)))
            {
                return ConvolutionMethod::FFT;
            }
            if(src->dimension(idx_c) < 16)
            {
                return ConvolutionMethod::GEMM;
            }
            return bool(ClWinogradConv2d::validate(src, weights, nullptr, dst, conv_info, act_info, enable_fast_math)) ? ConvolutionMethod::WINOGRAD : ConvolutionMethod::GEMM;
        }
        else
        {
            const bool   is_direct_valid           = bool(ClDirectConv2d::validate(src, weights, nullptr, dst, conv_info, act_info));
            const bool   is_wino_valid             = bool(ClWinogradConv2d::validate(src, weights, nullptr, dst, conv_info, act_info, enable_fast_math));
            const size_t kernel_sz_direct_conv_thr = get_direct_conv_kernel_threshold_nhwc(gpu_target);

            // SRGAN case
            if((src->dimension(idx_h) > 720U) && (dst->dimension(idx_h) > 720U) && (weights->dimension(idx_h) == 9) && (conv_info.pad_top() < 3)
               && is_direct_valid)
            {
                return ConvolutionMethod::DIRECT;
            }

            // Floating-point case: GeMM/Direct/Winograd
            if(is_data_type_float(src->data_type()))
            {
                // Get dst shape
                TensorShape output_shape       = misc::shape_calculator::compute_deep_convolution_shape(*src, *weights, conv_info);
                const bool  is_large_kernel_sz = (weights->dimension(idx_w) >= kernel_sz_direct_conv_thr) && (weights->dimension(idx_h) >= kernel_sz_direct_conv_thr);
                const bool  is_ifm_ge_8        = src->dimension(idx_c) >= 8;
                const bool  is_ifm_ge_16       = src->dimension(idx_c) >= 16;
                const bool  is_ofm_lte_8       = weights->dimension(3U) <= 8;
                const bool  is_ofm_lt_64       = weights->dimension(3U) < 64;
                const bool  workload_gte_8192  = (output_shape[0] * output_shape[1] * output_shape[2]) / 16 >= 8192;
                const bool  is_ifm_gt_ofm      = src->dimension(idx_c) > weights->dimension(3U);
                const bool  is_m_one           = output_shape[1] * output_shape[2] == 1;
                const bool  is_unit_stride     = (conv2d_info.conv_info.stride().first == 1) && (conv2d_info.conv_info.stride().second == 1);
                const int32_t kernel_sz        = weights->dimension(idx_w) * weights->dimension(idx_h);

                // Run Winograd if valid and IFM >= 8
                if(is_wino_valid && is_ifm_ge_8)
                {
                    if(is_ofm_lte_8)
                    {
                        if(gpu_target == arm_compute::GPUTarget::G71 || gpu_target == arm_compute::GPUTarget::G72 || get_arch_from_target(gpu_target) == arm_compute::GPUTarget::MIDGARD)
                        {
                            return ConvolutionMethod::WINOGRAD;
                        }
                    }
                    else
                    {
                        return ConvolutionMethod::WINOGRAD;
                    }
                }

                // Direct convolution case
                if(is_direct_valid)
                {
                    if((gpu_target == arm_compute::GPUTarget::G71 || gpu_target == arm_compute::GPUTarget::G72 || get_arch_from_target(gpu_target) == arm_compute::GPUTarget::MIDGARD))
                    {
                        if(is_large_kernel_sz && is_ifm_ge_16 && is_ifm_gt_ofm)
                        {
                            return ConvolutionMethod::DIRECT;
                        }
                    }
                    else if(gpu_target == arm_compute::GPUTarget::G76)
                    {
                        if((is_large_kernel_sz && workload_gte_8192 && is_ifm_ge_16) || (is_ofm_lte_8 && is_ifm_ge_16))
                        {
                            return ConvolutionMethod::DIRECT;
                        }
                    }
                    else
                    {
                        ConvolutionMethod preferred_conv_method = ConvolutionMethod::DIRECT;

                        const bool is_indirect_valid = bool(ClIndirectConv2d::validate(src, weights, nullptr, dst, conv_info, act_info));

                        // indirect conv2d should be called when:
                        // 1- When the kernel size is greater than 1x1 and less than or equal to 9x9 (81)
                        // 2- When the kernel size is odd
                        // 3- When the Gpu target is Arm Mali-G77
                        if(is_indirect_valid)
                        {
                            const bool is_kernel_sz_odd = kernel_sz % 2;
                            const bool is_g77           = gpu_target == GPUTarget::G77;
                            preferred_conv_method = (kernel_sz > 1) && (kernel_sz <= 81) && is_kernel_sz_odd && is_g77? ConvolutionMethod::INDIRECT : ConvolutionMethod::DIRECT;
                        }

                        // Direct/indirect convolution used for the first layer of the network
                        if(workload_gte_8192 && !is_ifm_ge_16 && !is_unit_stride && is_ofm_lt_64)
                        {
                            // In general, the question we should ask for the first convolution layer of a model is:
                            // when the execution time of im2col + gemm < direct?. Since im2col does not depend on the OFM, it means that
                            // when OFM is big enough, the contribution of im2col is small and the GEMM approach is preferable.
                            // From internal experiments, the OFM threshold is 64 (is_ofm_lt_64)
                            return preferred_conv_method;
                        }

                        if((is_large_kernel_sz || is_m_one) && workload_gte_8192 && is_ifm_ge_16)
                        {
                            return preferred_conv_method;
                        }

                        // Direct convolution used for the last layer of the network
                        if(is_ofm_lte_8)
                        {
                            return preferred_conv_method;
                        }
                    }
                }

                // Default case
                return ConvolutionMethod::GEMM;
            }

            // Generic case for quantized. Only GeMM
            return ConvolutionMethod::GEMM;
        }
    }
}

void ClConv2d::run(ITensorPack &tensors)
{
    prepare(tensors);
    _operator->run(tensors);
}

void ClConv2d::prepare(ITensorPack &tensors)
{
    _operator->prepare(tensors);
}

experimental::MemoryRequirements ClConv2d::workspace() const
{
    return _aux_mem;
}
} // namespace opencl
} // namespace arm_compute
