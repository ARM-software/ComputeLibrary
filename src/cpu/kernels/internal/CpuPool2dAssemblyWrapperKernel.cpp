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
#include "src/cpu/kernels/internal/CpuPool2dAssemblyWrapperKernel.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/INEKernel.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
using namespace arm_compute::misc::shape_calculator;

void CpuPool2dAssemblyWrapperKernel::configure(const ITensorInfo *src, ITensorInfo *dst, const PoolingLayerInfo &info, const CPUInfo &cpu_info)
{
    ARM_COMPUTE_UNUSED(cpu_info);
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    // dst initialization if not yet initialized
    auto_init_if_empty(*dst, src->clone()->set_tensor_shape(compute_pool_shape(*src, info)));

#if defined(__aarch64__)
    const bool requantize = src->quantization_info() != dst->quantization_info();

    switch(src->data_type())
    {
        case DataType::QASYMM8:
            if(requantize)
            {
                create_arm_pooling_requant<uint8_t, uint8_t>(src, dst, info, cpu_info);
            }
            else
            {
                create_arm_pooling<uint8_t, uint8_t>(src, dst, info, cpu_info);
            }
            break;
        case DataType::QASYMM8_SIGNED:
            if(requantize)
            {
                create_arm_pooling_requant<int8_t, int8_t>(src, dst, info, cpu_info);
            }
            else
            {
                create_arm_pooling<int8_t, int8_t>(src, dst, info, cpu_info);
            }
            break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            create_arm_pooling<float16_t, float16_t>(src, dst, info, cpu_info);
            break;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        case DataType::F32:
            create_arm_pooling<float, float>(src, dst, info, cpu_info);
            break;
        default:
            break;
    }
#endif // defined(__aarch64__)

    Window win = calculate_max_window(*dst, Steps());
    INEKernel::configure(win);
}

Status CpuPool2dAssemblyWrapperKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, const PoolingLayerInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);

#ifndef __aarch64__
    ARM_COMPUTE_RETURN_ERROR_MSG("32-bit is not supported by assembly kernels");
#endif /* __aarch64__ */
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((src->data_layout() != DataLayout::NHWC) || (info.data_layout != DataLayout::NHWC), "Only NHWC is supported by assembly kernels");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((info.pool_type != PoolingType::AVG) && (info.pool_type != PoolingType::MAX),
                                    "Only AVG and MAX pooling are supported by assembly kernels");

    if(dst->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);

        const auto src_qinfo = src->quantization_info().uniform();
        const auto dst_qinfo = dst->quantization_info().uniform();

        if(src_qinfo != dst_qinfo)
        {
            const float multiplier = src_qinfo.scale / dst_qinfo.scale;
            int32_t     dst_multiplier{};
            int32_t     dst_shift{};
            ARM_COMPUTE_RETURN_ERROR_ON(quantization::calculate_quantized_multiplier(multiplier, &dst_multiplier, &dst_shift));
        }
        else
        {
            if(src->data_type() == DataType::QASYMM8)
            {
                const bool has_padding = info.pad_stride_info.has_padding();
                ARM_COMPUTE_RETURN_ERROR_ON_MSG(!info.exclude_padding && has_padding, "Assembly kernels do not support padding for QASYMM8 with same src/dst quantization info");
            }
        }
    }
    else
    {
        if(src->data_type() == DataType::QASYMM8)
        {
            // If dst is not configured, the quantization info are the same
            const bool has_padding = info.pad_stride_info.has_padding();
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(!info.exclude_padding && has_padding, "Assembly kernels do not support padding for QASYMM8 with same src/dst quantization info");
        }
    }
    return Status{};
}

void CpuPool2dAssemblyWrapperKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(_kernel_asm.get());
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_UNUSED(window);
    ARM_COMPUTE_UNUSED(info);

    ARM_COMPUTE_ERROR_ON(tensors.empty());

    const ITensor *src       = tensors.get_const_tensor(TensorType::ACL_SRC);
    ITensor       *dst       = tensors.get_tensor(TensorType::ACL_DST);
    ITensor       *workspace = tensors.get_tensor(TensorType::ACL_INT_0);

    const auto in_ptr        = src->buffer() + src->info()->offset_first_element_in_bytes();
    auto       out_ptr       = dst->buffer() + dst->info()->offset_first_element_in_bytes();
    auto       working_space = workspace->buffer() + workspace->info()->offset_first_element_in_bytes();

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

    _kernel_asm->execute(in_ptr, ld_src_col, ld_src_row, ld_src_batch,
                         out_ptr, ld_dst_col, ld_dst_row, ld_dst_batch,
                         working_space, info.thread_id, info.num_threads);
}

size_t CpuPool2dAssemblyWrapperKernel::get_working_size(unsigned int num_threads) const
{
    return _kernel_asm->get_working_size(num_threads);
}

bool CpuPool2dAssemblyWrapperKernel::is_configured() const
{
    return _kernel_asm != nullptr;
}

template <typename Typesrc, typename Typedst>
void CpuPool2dAssemblyWrapperKernel::create_arm_pooling(const ITensorInfo *src, ITensorInfo *dst, const PoolingLayerInfo &info, const CPUInfo &cpu_info)
{
    const arm_conv::pooling::PoolingType pool_type = (info.pool_type == PoolingType::AVG) ? arm_conv::pooling::PoolingType::AVERAGE : arm_conv::pooling::PoolingType::MAX;

    arm_conv::pooling::PoolingWindow window{};
    window.cols = static_cast<unsigned int>(info.pool_size.x());
    window.rows = static_cast<unsigned int>(info.pool_size.y());

    arm_conv::pooling::PoolingStride stride{};
    std::tie(stride.cols, stride.rows) = info.pad_stride_info.stride();

    const arm_conv::PaddingValues padding{ info.pad_stride_info.pad_left(), info.pad_stride_info.pad_top(), info.pad_stride_info.pad_right(), info.pad_stride_info.pad_bottom() };

    constexpr unsigned int idx_width    = 1;
    constexpr unsigned int idx_height   = 2;
    constexpr unsigned int idx_channels = 0;
    constexpr unsigned int idx_batches  = 3;

    const unsigned int n_batches  = src->dimension(idx_batches);
    const unsigned int src_rows   = src->dimension(idx_height);
    const unsigned int src_cols   = src->dimension(idx_width);
    const unsigned int n_channels = src->dimension(idx_channels);
    const unsigned int dst_rows   = dst->dimension(idx_height);
    const unsigned int dst_cols   = dst->dimension(idx_width);

    arm_conv::pooling::PoolingArgs args(&cpu_info, pool_type, window, stride, info.exclude_padding, n_batches, src_rows, src_cols, n_channels, dst_rows, dst_cols, padding, nullptr);

    // Configure assembly pooling kernel
    auto pooling_kernel_asm = arm_conv::pooling::pooling<Typesrc, Typedst>(args);
    if(pooling_kernel_asm == nullptr)
    {
        // Configuration not supported: Leave function unconfigured:
        return;
    }

    _kernel_asm = std::move(pooling_kernel_asm);
}

template <typename Typesrc, typename Typedst>
void CpuPool2dAssemblyWrapperKernel::create_arm_pooling_requant(const ITensorInfo *src, ITensorInfo *dst, const PoolingLayerInfo &info, const CPUInfo &cpu_info)
{
    const arm_conv::pooling::PoolingType pool_type = (info.pool_type == PoolingType::AVG) ? arm_conv::pooling::PoolingType::AVERAGE : arm_conv::pooling::PoolingType::MAX;

    arm_conv::pooling::PoolingWindow window{};
    window.cols = static_cast<unsigned int>(info.pool_size.x());
    window.rows = static_cast<unsigned int>(info.pool_size.y());

    arm_conv::pooling::PoolingStride stride{};
    std::tie(stride.cols, stride.rows) = info.pad_stride_info.stride();

    const arm_conv::PaddingValues padding{ info.pad_stride_info.pad_left(), info.pad_stride_info.pad_top(), info.pad_stride_info.pad_right(), info.pad_stride_info.pad_bottom() };

    constexpr unsigned int idx_width    = 1;
    constexpr unsigned int idx_height   = 2;
    constexpr unsigned int idx_channels = 0;
    constexpr unsigned int idx_batches  = 3;

    const unsigned int n_batches  = src->dimension(idx_batches);
    const unsigned int src_rows   = src->dimension(idx_height);
    const unsigned int src_cols   = src->dimension(idx_width);
    const unsigned int n_channels = src->dimension(idx_channels);
    const unsigned int dst_rows   = dst->dimension(idx_height);
    const unsigned int dst_cols   = dst->dimension(idx_width);

    arm_conv::pooling::PoolingArgs args(&cpu_info, pool_type, window, stride, info.exclude_padding, n_batches, src_rows, src_cols, n_channels, dst_rows, dst_cols, padding, nullptr);

    const auto src_qinfo = src->quantization_info().uniform();
    const auto dst_qinfo = dst->quantization_info().uniform();

    const float multiplier = src_qinfo.scale / dst_qinfo.scale;
    int32_t     dst_multiplier{};
    int32_t     dst_shift{};
    quantization::calculate_quantized_multiplier(multiplier, &dst_multiplier, &dst_shift);

    const arm_conv::pooling::Requantize32 requant_args(src_qinfo.offset,
                                                       dst_qinfo.offset,
                                                       dst_shift, // left shift
                                                       0,         // right shift
                                                       dst_multiplier);

    // Configure assembly pooling kernel with requantization
    auto pooling_kernel_asm = arm_conv::pooling::pooling<Typesrc, Typedst, arm_conv::pooling::Requantize32>(args, requant_args);
    if(pooling_kernel_asm == nullptr)
    {
        // Configuration not supported: Leave function unconfigured:
        return;
    }

    _kernel_asm = std::move(pooling_kernel_asm);
}

size_t CpuPool2dAssemblyWrapperKernel::get_mws(const CPUInfo &platform, size_t thread_count) const
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
