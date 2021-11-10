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
#include "src/cpu/kernels/internal/CpuDepthwiseConv2dAssemblyWrapperKernel.h"

#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/utils/AssemblyUtils.h"

#include "src/core/NEON/kernels/assembly/depthwise.hpp"

#include "depthwise_common.hpp"

#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
using namespace arm_compute::misc::shape_calculator;

namespace
{
constexpr unsigned int idx_width    = 1;
constexpr unsigned int idx_height   = 2;
constexpr unsigned int idx_channels = 0;
constexpr unsigned int idx_batches  = 3;

template <typename TSrc, typename TWeights, typename TDst>
void create_arm_dwc(const ITensorInfo *src, const ITensorInfo *weights, ITensorInfo *dst,
                    const ConvolutionInfo &info, const CPUInfo &cpu_info,
                    std::unique_ptr<arm_conv::depthwise::IDepthwiseCommon> &kernel)
{
    unsigned int stride_cols{};
    unsigned int stride_rows{};
    std::tie(stride_cols, stride_rows) = info.pad_stride_info.stride();

    const arm_conv::PaddingValues padding = assembly_utils::map_to_arm_conv_padding(info.pad_stride_info);

    const unsigned int n_batches  = src->dimension(idx_batches);
    const unsigned int src_rows   = src->dimension(idx_height);
    const unsigned int src_cols   = src->dimension(idx_width);
    const unsigned int n_channels = src->dimension(idx_channels);
    const unsigned int dst_rows   = dst->dimension(idx_height);
    const unsigned int dst_cols   = dst->dimension(idx_width);

    const unsigned int kernel_cols = weights->dimension(idx_width);
    const unsigned int kernel_rows = weights->dimension(idx_height);

    const arm_gemm::Activation activation = assembly_utils::map_to_arm_gemm_activation(info.act_info);

    arm_conv::depthwise::DepthwiseArgs args(&cpu_info, kernel_rows, kernel_cols, stride_rows, stride_cols,
                                            n_batches, src_rows, src_cols, n_channels, dst_rows, dst_cols, info.depth_multiplier,
                                            padding, activation, nullptr);

    // Configure assembly pooling kernel
    auto dwc_kernel_asm = arm_conv::depthwise::depthwise<TSrc, TWeights, TDst>(args);
    if(dwc_kernel_asm == nullptr)
    {
        // Configuration not supported: Leave function unconfigured:
        return;
    }

    kernel = std::move(dwc_kernel_asm);
}

template <typename TSrc, typename TWeights, typename TDst>
void create_arm_dwc_quant(const ITensorInfo *src, const ITensorInfo *weights, ITensorInfo *dst,
                          const ConvolutionInfo &info, const CPUInfo &cpu_info,
                          std::unique_ptr<arm_conv::depthwise::IDepthwiseCommon> &kernel,
                          std::vector<int32_t> &multipliers, std::vector<int32_t> &right_shifts, std::vector<int32_t> &left_shifts)
{
    unsigned int stride_cols{};
    unsigned int stride_rows{};
    std::tie(stride_cols, stride_rows) = info.pad_stride_info.stride();

    const arm_conv::PaddingValues padding = assembly_utils::map_to_arm_conv_padding(info.pad_stride_info);

    const unsigned int n_batches  = src->dimension(idx_batches);
    const unsigned int src_rows   = src->dimension(idx_height);
    const unsigned int src_cols   = src->dimension(idx_width);
    const unsigned int n_channels = src->dimension(idx_channels);
    const unsigned int dst_rows   = dst->dimension(idx_height);
    const unsigned int dst_cols   = dst->dimension(idx_width);

    const unsigned int kernel_cols = weights->dimension(idx_width);
    const unsigned int kernel_rows = weights->dimension(idx_height);

    const arm_gemm::Activation activation = assembly_utils::map_to_arm_gemm_activation(info.act_info);

    arm_conv::depthwise::DepthwiseArgs args(&cpu_info, kernel_rows, kernel_cols, stride_rows, stride_cols,
                                            n_batches, src_rows, src_cols, n_channels, dst_rows, dst_cols, info.depth_multiplier,
                                            padding, activation, nullptr);

    const auto src_qinfo     = src->quantization_info().uniform();
    const auto weights_qinfo = weights->quantization_info();
    const auto dst_qinfo     = dst->quantization_info().uniform();

    const unsigned int num_filters = weights_qinfo.scale().size();

    multipliers.resize(num_filters);
    std::vector<int32_t> dst_shifts(num_filters);
    quantization::compute_quantized_multipliers_and_shifts(src,
                                                           weights,
                                                           dst,
                                                           multipliers.data(),
                                                           dst_shifts.data());

    // Quantize activation bounds
    int32_t min_activation = std::numeric_limits<TSrc>::lowest();
    int32_t max_activation = std::numeric_limits<TSrc>::max();
    if(info.act_info.enabled())
    {
        std::tie(min_activation, max_activation) = get_quantized_activation_min_max(info.act_info, src->data_type(), dst_qinfo);
    }

    // Set quantization parameters for assembly kernels
    arm_gemm::Requantize32 requant_args{};
    if(is_data_type_quantized_per_channel(weights->data_type()))
    {
        left_shifts.resize(num_filters);
        right_shifts.resize(num_filters);
        bool need_left_shift = false; // Select more optimized path if left shift is not needed
        for(unsigned int i = 0; i < num_filters; ++i)
        {
            left_shifts[i]  = std::max(-dst_shifts[i], static_cast<int32_t>(0));
            right_shifts[i] = std::min(-dst_shifts[i], static_cast<int32_t>(0));
            if(dst_shifts[i] < 0 && !need_left_shift)
            {
                need_left_shift = true;
            }
        }

        requant_args = arm_gemm::Requantize32(nullptr,
                                              0,
                                              src_qinfo.offset,
                                              weights_qinfo.uniform().offset,
                                              dst_qinfo.offset,
                                              (need_left_shift) ? left_shifts.data() : nullptr,
                                              right_shifts.data(),
                                              multipliers.data(),
                                              static_cast<TSrc>(min_activation),
                                              static_cast<TSrc>(max_activation));
    }
    else
    {
        requant_args = arm_gemm::Requantize32(nullptr,
                                              0,
                                              src_qinfo.offset,
                                              weights_qinfo.uniform().offset,
                                              dst_qinfo.offset,
                                              -dst_shifts[0],
                                              multipliers[0],
                                              static_cast<TSrc>(min_activation),
                                              static_cast<TSrc>(max_activation));
    }

    // Configure assembly pooling kernel with requantization
    auto dwc_kernel_asm = arm_conv::depthwise::depthwise<TSrc, TWeights, TDst, arm_gemm::Requantize32>(args, requant_args);
    if(dwc_kernel_asm == nullptr)
    {
        // Configuration not supported: Leave function unconfigured:
        return;
    }

    kernel = std::move(dwc_kernel_asm);
}
} // namespace

CpuDepthwiseConv2dAssemblyWrapperKernel::CpuDepthwiseConv2dAssemblyWrapperKernel()
    : _kernel_asm(nullptr),
      _multipliers(),
      _left_shifts(),
      _right_shifts()
{
}

CpuDepthwiseConv2dAssemblyWrapperKernel::~CpuDepthwiseConv2dAssemblyWrapperKernel() = default;

void CpuDepthwiseConv2dAssemblyWrapperKernel::configure(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *, ITensorInfo *dst,
                                                        const ConvolutionInfo &info, const CPUInfo &cpu_info)
{
    ARM_COMPUTE_UNUSED(cpu_info);
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights, dst);

    // Destination initialization if not yet initialized
    const TensorShape dst_shape = compute_depthwise_convolution_shape(*src, *weights, info);
    auto_init_if_empty(*dst, src->clone()->set_tensor_shape(dst_shape));

#if defined(__aarch64__)
    switch(src->data_type())
    {
        case DataType::QASYMM8:
            if(is_data_type_quantized_per_channel(weights->data_type()))
            {
                create_arm_dwc_quant<uint8_t, int8_t, uint8_t>(src, weights, dst, info, cpu_info, _kernel_asm, _multipliers, _right_shifts, _left_shifts);
            }
            else
            {
                create_arm_dwc_quant<uint8_t, uint8_t, uint8_t>(src, weights, dst, info, cpu_info, _kernel_asm, _multipliers, _right_shifts, _left_shifts);
            }
            break;
        case DataType::QASYMM8_SIGNED:
            create_arm_dwc_quant<int8_t, int8_t, int8_t>(src, weights, dst, info, cpu_info, _kernel_asm, _multipliers, _right_shifts, _left_shifts);
            break;
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        case DataType::F16:
            create_arm_dwc<float16_t, float16_t, float16_t>(src, weights, dst, info, cpu_info, _kernel_asm);
            break;
#endif // defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        case DataType::F32:
            create_arm_dwc<float, float, float>(src, weights, dst, info, cpu_info, _kernel_asm);
            break;
        default:
            break;
    }
#endif // defined(__aarch64__)

    Window win = calculate_max_window(*dst, Steps());
    ICpuKernel::configure(win);
}

Status CpuDepthwiseConv2dAssemblyWrapperKernel::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *bias, const ITensorInfo *dst, const ConvolutionInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);

#if !defined(__aarch64__)
    ARM_COMPUTE_RETURN_ERROR_MSG("32-bit is not supported by assembly kernels");
#endif // !defined(__aarch64__)
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(src->data_layout() != DataLayout::NHWC, "Only NHWC is supported by assembly kernels");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(info.dilation != Size2D(1, 1), "Assembly kernels do not support dilation != (1, 1)");

    if(is_data_type_quantized_per_channel(weights->data_type()))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QSYMM8_PER_CHANNEL);
        ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(0) != weights->quantization_info().scale().size());
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, weights);
    }

    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(bias->dimension(0) != weights->dimension(0));

        if(is_data_type_quantized(src->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(bias, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, bias);
        }
    }

    if(dst->total_size() > 0)
    {
        const TensorShape dst_shape = misc::shape_calculator::compute_depthwise_convolution_shape(*src, *weights, info);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(), dst_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    }
    return Status{};
}

void CpuDepthwiseConv2dAssemblyWrapperKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(_kernel_asm.get());
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_UNUSED(info);

    ARM_COMPUTE_ERROR_ON(tensors.empty());

    const ITensor *src       = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    ITensor       *dst       = tensors.get_tensor(TensorType::ACL_DST);
    ITensor       *workspace = tensors.get_tensor(TensorType::ACL_INT_0);
    ITensor       *storage   = tensors.get_tensor(TensorType::ACL_INT_1);

    const auto src_ptr        = src->buffer() + src->info()->offset_first_element_in_bytes();
    auto       dst_ptr        = dst->buffer() + dst->info()->offset_first_element_in_bytes();
    auto       working_space  = workspace->buffer() + workspace->info()->offset_first_element_in_bytes();
    auto       parameters_ptr = storage->buffer() + storage->info()->offset_first_element_in_bytes();

    const auto src_shape   = src->info()->tensor_shape();
    const auto dst_shape   = dst->info()->tensor_shape();
    const auto src_padding = src->info()->padding();
    const auto dst_padding = dst->info()->padding();

    const size_t ld_src_col   = src_shape[0] + src_padding.left + src_padding.right;
    const size_t ld_src_row   = ld_src_col * (src_shape[1] + src_padding.top + src_padding.bottom);
    const size_t ld_src_batch = ld_src_row * src_shape[2];
    const size_t ld_dst_col   = dst_shape[0] + dst_padding.left + dst_padding.right;
    const size_t ld_dst_row   = ld_dst_col * (dst_shape[1] + dst_padding.top + dst_padding.bottom);
    const size_t ld_dst_batch = ld_dst_row * dst_shape[2];

    _kernel_asm->execute(src_ptr, ld_src_col, ld_src_row, ld_src_batch,
                         parameters_ptr,
                         dst_ptr, ld_dst_col, ld_dst_row, ld_dst_batch,
                         working_space, info.thread_id, info.num_threads);
}

void CpuDepthwiseConv2dAssemblyWrapperKernel::pack_parameters(void *parameters_ptr, void *bias_ptr, void *weights_ptr, size_t ld_weights_col, size_t ld_weight_row)
{
    _kernel_asm->pack_parameters(parameters_ptr, bias_ptr, weights_ptr, ld_weights_col, ld_weight_row);
}

size_t CpuDepthwiseConv2dAssemblyWrapperKernel::get_storage_size() const
{
    return _kernel_asm->get_storage_size();
}

size_t CpuDepthwiseConv2dAssemblyWrapperKernel::get_working_size(unsigned int num_threads, unsigned int num_input_channels) const
{
    return _kernel_asm->get_working_size(num_threads, num_input_channels);
}

bool CpuDepthwiseConv2dAssemblyWrapperKernel::is_configured() const
{
    return _kernel_asm != nullptr;
}

const char *CpuDepthwiseConv2dAssemblyWrapperKernel::name() const
{
    return "CpuDepthwiseConv2dAssemblyWrapperKernel";
}

size_t CpuDepthwiseConv2dAssemblyWrapperKernel::get_mws(const CPUInfo &platform, size_t thread_count) const
{
    ARM_COMPUTE_UNUSED(thread_count);
    // Tuning results that gave optimized results in performance investigation 
    if (platform.get_cpu_model() == CPUModel::A73 ) 
    {
        return 10240;
    }
    else 
    {
        return 9216;
    }
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
