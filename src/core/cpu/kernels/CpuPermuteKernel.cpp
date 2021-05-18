/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "src/core/cpu/kernels/CpuPermuteKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

namespace
{
#include "src/core/NEON/kernels/convolution/common/shims.hpp"
} // namespace

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
inline bool is_permutation_supported(const PermutationVector &v)
{
    static const std::array<PermutationVector, 2> permutations2 =
    {
        {
            PermutationVector(0U, 1U),
            PermutationVector(1U, 0U),
        }
    };
    static const std::array<PermutationVector, 6> permutations3 =
    {
        {
            PermutationVector(2U, 0U, 1U),
            PermutationVector(1U, 2U, 0U),
            PermutationVector(0U, 1U, 2U),
            PermutationVector(0U, 2U, 1U),
            PermutationVector(1U, 0U, 2U),
            PermutationVector(2U, 1U, 0U),
        }
    };
    static const std::array<PermutationVector, 24> permutations4 =
    {
        {
            PermutationVector(0U, 1U, 2U, 3U),
            PermutationVector(1U, 0U, 2U, 3U),
            PermutationVector(2U, 0U, 1U, 3U),
            PermutationVector(0U, 2U, 1U, 3U),
            PermutationVector(1U, 2U, 0U, 3U),
            PermutationVector(2U, 1U, 0U, 3U),
            PermutationVector(2U, 1U, 3U, 0U),
            PermutationVector(1U, 2U, 3U, 0U),
            PermutationVector(3U, 2U, 1U, 0U),
            PermutationVector(2U, 3U, 1U, 0U),
            PermutationVector(1U, 3U, 2U, 0U),
            PermutationVector(3U, 1U, 2U, 0U),
            PermutationVector(3U, 0U, 2U, 1U),
            PermutationVector(0U, 3U, 2U, 1U),
            PermutationVector(2U, 3U, 0U, 1U),
            PermutationVector(3U, 2U, 0U, 1U),
            PermutationVector(0U, 2U, 3U, 1U),
            PermutationVector(2U, 0U, 3U, 1U),
            PermutationVector(1U, 0U, 3U, 2U),
            PermutationVector(0U, 1U, 3U, 2U),
            PermutationVector(3U, 1U, 0U, 2U),
            PermutationVector(1U, 3U, 0U, 2U),
            PermutationVector(0U, 3U, 1U, 2U),
            PermutationVector(3U, 0U, 1U, 2U)
        }
    };

    return (permutations2.end() != std::find(permutations2.begin(), permutations2.end(), v)) || (permutations3.end() != std::find(permutations3.begin(), permutations3.end(), v))
           || (permutations4.end() != std::find(permutations4.begin(), permutations4.end(), v));
}

Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst, const PermutationVector &perm)
{
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!is_permutation_supported(perm), "PermutationVector not supported.");

    const TensorShape dst_shape = misc::shape_calculator::compute_permutation_output_shape(*src, perm);

    // Validate configured destination
    if(dst->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(), dst_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    }

    return Status{};
}

template <typename T>
void run_permute(const Window &window, const ITensor *src, const ITensor *dst, const PermutationVector &perm)
{
    const DataLayout src_layout = src->info()->data_layout();

    // Source window
    Window window_src = window;

    // we only support these two configs in src/core/NEON/kernels/convolution/common/shims.hpp, for all others
    // we have to fall back to C++
    if((src_layout == DataLayout::NCHW && perm == PermutationVector{ 2U, 0U, 1U }) || (src_layout == DataLayout::NHWC && perm == PermutationVector{ 1U, 2U, 0U }))
    {
        window_src.set(Window::DimX, Window::Dimension(window.x().start(), window.x().end(), window.x().end() - window.x().start()));
        window_src.set(Window::DimY, Window::Dimension(window.y().start(), window.y().end(), window.y().end() - window.y().start()));
        window_src.set(Window::DimZ, Window::Dimension(window.z().start(), window.z().end(), window.z().end() - window.z().start()));
        window_src.set(3, Window::Dimension(window[3].start(), window[3].end(), window[3].end() - window[3].start()));
    }

    // Destination window
    Window                  window_dst(window);
    const Window::Dimension zero_window = Window::Dimension(0, 0, 0);
    for(size_t d = 0; d <= dst->info()->num_dimensions(); ++d)
    {
        window_dst.set(d, zero_window);
    }

    // Create iterators
    Iterator src_it(src, window_src);
    Iterator dst_it(dst, window_dst);

    int in_row_stride     = 0;
    int in_col_stride     = 0;
    int in_channel_stride = 0;
    int in_batch_stride   = 0;
    int n_cols            = 0;
    int n_rows            = 0;
    int n_channels        = 0;
    int n_batches         = 0;

    switch(src_layout)
    {
        case DataLayout::NCHW:
        {
            in_row_stride     = src->info()->strides_in_bytes().y() / sizeof(T);
            in_channel_stride = src->info()->strides_in_bytes().z() / sizeof(T);
            in_batch_stride   = src->info()->strides_in_bytes()[3] / sizeof(T);
            n_cols            = src->info()->tensor_shape().x();
            n_rows            = window_src.y().step();
            n_channels        = src->info()->tensor_shape().z();
            n_batches         = src->info()->tensor_shape()[3];
            break;
        }
        case DataLayout::NHWC:
        {
            in_col_stride   = src->info()->strides_in_bytes().y() / sizeof(T);
            in_row_stride   = src->info()->strides_in_bytes().z() / sizeof(T);
            in_batch_stride = src->info()->strides_in_bytes()[3] / sizeof(T);
            n_channels      = src->info()->tensor_shape().x();
            n_cols          = window_src.y().step();
            n_rows          = src->info()->tensor_shape().z();
            n_batches       = src->info()->tensor_shape()[3];
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Invalid source data layout.");
            break;
        }
    }

    // CHW -> HWC
    if(src_layout == DataLayout::NCHW && perm == PermutationVector{ 2U, 0U, 1U })
    {
        const int out_channel_stride = dst->info()->strides_in_bytes().x() / sizeof(T);
        const int out_col_stride     = dst->info()->strides_in_bytes().y() / sizeof(T);
        const int out_row_stride     = dst->info()->strides_in_bytes().z() / sizeof(T);
        const int out_batch_stride   = dst->info()->strides_in_bytes()[3] / sizeof(T);
        execute_window_loop(window_src, [&](const Coordinates & id)
        {
            const int idx = id[0] * out_col_stride + id[1] * out_row_stride + id[2] * out_channel_stride;
            reorder::nchw_to_nhwc(reinterpret_cast<const T *>(src_it.ptr()), reinterpret_cast<T *>(dst_it.ptr()) + idx,
                                  n_batches, n_channels, n_rows, n_cols,
                                  in_batch_stride, in_channel_stride, in_row_stride,
                                  out_batch_stride, out_row_stride, out_col_stride);
        },
        src_it, dst_it);
    }
    // HWC -> CHW
    else if(src_layout == DataLayout::NHWC && perm == PermutationVector{ 1U, 2U, 0U })
    {
        const int out_col_stride     = dst->info()->strides_in_bytes().x() / sizeof(T);
        const int out_row_stride     = dst->info()->strides_in_bytes().y() / sizeof(T);
        const int out_channel_stride = dst->info()->strides_in_bytes().z() / sizeof(T);
        const int out_batch_stride   = dst->info()->strides_in_bytes()[3] / sizeof(T);
        execute_window_loop(window_src, [&](const Coordinates & id)
        {
            const int idx = id[0] * out_channel_stride + id[1] * out_col_stride + id[2] * out_row_stride;
            reorder::nhwc_to_nchw(reinterpret_cast<const T *>(src_it.ptr()), reinterpret_cast<T *>(dst_it.ptr()) + idx,
                                  n_batches, n_rows, n_cols, n_channels,
                                  in_batch_stride, in_row_stride, in_col_stride,
                                  out_batch_stride, out_channel_stride, out_row_stride);
        },
        src_it, dst_it);
    }
    else
    {
        // All other cases fall back to C++
        // Permute strides
        Strides strides      = dst->info()->strides_in_bytes();
        Strides perm_strides = strides;
        permute_strides(perm_strides, perm);
        const int perm_stride_3 = src->info()->num_dimensions() >= 4 ? perm_strides[3] : 0;
        execute_window_loop(window, [&](const Coordinates & id)
        {
            const int idx                                = id[0] * perm_strides[0] + id[1] * perm_strides[1] + id[2] * perm_strides[2] + id[3] * perm_stride_3;
            *(reinterpret_cast<T *>(dst_it.ptr() + idx)) = *(reinterpret_cast<const T *>(src_it.ptr()));
        },
        src_it, dst_it);
    }
}
} // namespace

void CpuPermuteKernel::configure(const ITensorInfo *src, ITensorInfo *dst, const PermutationVector &perm)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    const TensorShape dst_shape = misc::shape_calculator::compute_permutation_output_shape(*src, perm);
    // Destination auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, src->clone()->set_tensor_shape(dst_shape));

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst, perm));

    _perm = perm;

    // Configure kernel window
    Window win = calculate_max_window(*src, Steps());

    // This kernel doesn't need padding so update_window_and_padding() can be skipped

    ICpuKernel::configure(win);
}

Status CpuPermuteKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, const PermutationVector &perm)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst, perm));
    return Status{};
}

void CpuPermuteKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    const auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto       dst = tensors.get_tensor(TensorType::ACL_DST);

    switch(src->info()->element_size())
    {
        case 1:
            run_permute<uint8_t>(window, src, dst, _perm);
            break;
        case 2:
            run_permute<uint16_t>(window, src, dst, _perm);
            break;
        case 4:
            run_permute<uint32_t>(window, src, dst, _perm);
            break;
        default:
            ARM_COMPUTE_ERROR("Element size not supported");
            break;
    }
}

const char *CpuPermuteKernel::name() const
{
    return "CpuPermuteKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
