/*
 * Copyright (c) 2017-2023 Arm Limited.
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
#include "src/cpu/kernels/CpuReshapeKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include "src/core/helpers/Utils.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/INEKernel.h"

#include <cstdint>

/** [NEReshapeLayerKernel Kernel] **/
namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    // Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src) is not needed here as this kernel doesn't use CPU FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_type() == DataType::UNKNOWN);

    if (dst->tensor_shape().total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON(src->tensor_shape().total_size() != dst->tensor_shape().total_size());
    }

    return Status{};
}

template <typename T>
void reshape_tensor_per_element(const Window &window, const ITensor *src, ITensor *dst)
{
    const TensorShape &src_shape = src->info()->tensor_shape();
    const TensorShape &dst_shape = dst->info()->tensor_shape();

    Iterator dst_it(dst, window);

    execute_window_loop(
        window,
        [&](const Coordinates &dst_coord)
        {
            Coordinates src_coord  = index2coords(src_shape, coords2index(dst_shape, dst_coord));
            const auto  output_ptr = dst->ptr_to_element(dst_coord);
            const auto  input_ptr  = src->ptr_to_element(src_coord);

            *reinterpret_cast<T *>(output_ptr) = *reinterpret_cast<T *>(input_ptr);
        },
        dst_it);
}

void reshape_tensor_per_element_selector(const Window &window, const ITensor *src, ITensor *dst)
{
    switch (src->info()->data_type())
    {
        case DataType::U8:
        case DataType::S8:
        case DataType::QSYMM8:
        case DataType::QASYMM8:
        case DataType::QASYMM8_SIGNED:
        case DataType::QSYMM8_PER_CHANNEL:
            reshape_tensor_per_element<uint8_t>(window, src, dst);
            break;
        case DataType::U16:
        case DataType::S16:
        case DataType::F16:
            reshape_tensor_per_element<uint16_t>(window, src, dst);
            break;
        case DataType::U32:
        case DataType::S32:
        case DataType::F32:
            reshape_tensor_per_element<uint32_t>(window, src, dst);
            break;
        case DataType::U64:
        case DataType::S64:
        case DataType::F64:
            reshape_tensor_per_element<uint64_t>(window, src, dst);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type!");
    }
}

void reshape_tensor_per_row(const Window &window, const ITensor *src, ITensor *dst)
{
    const TensorShape &src_shape = src->info()->tensor_shape();
    const TensorShape &dst_shape = dst->info()->tensor_shape();
    Coordinates        src_coord{};
    Coordinates        dst_coord{};

    const auto element_size      = dst->info()->element_size();
    const auto window_start_x    = static_cast<int>(window.x().start());
    const auto window_end_x      = static_cast<int>(window.x().end());
    const auto src_row_size      = static_cast<int>(src_shape[0]);
    const auto row_size_in_bytes = src_row_size * element_size;

    auto output_ptr = dst->ptr_to_element(dst_coord);
    auto input_ptr  = src->ptr_to_element(src_coord);

    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator dst_it(dst, win);
    execute_window_loop(
        win,
        [&](Coordinates &id)
        {
            dst_coord = id;

            for (int x = window_start_x; x < window_end_x; x += src_row_size)
            {
                src_coord  = index2coords(src_shape, coords2index(dst_shape, dst_coord));
                output_ptr = dst->ptr_to_element(dst_coord);
                input_ptr  = src->ptr_to_element(src_coord);

                std::memcpy(output_ptr, input_ptr, row_size_in_bytes);

                dst_coord.increment(Window::DimX, src_row_size);
            }
        },
        dst_it);
}

void reshape_tensor_per_window(const Window &window, const ITensor *src, ITensor *dst)
{
    Iterator src_it(src, window);
    Iterator dst_it(dst, window);

    const size_t element_size         = dst->info()->element_size();
    const auto   window_size          = window.x().end() - window.x().start();
    const auto   window_size_in_bytes = window_size * element_size;

    const auto input_ptr  = src_it.ptr();
    const auto output_ptr = dst_it.ptr();

    std::memcpy(output_ptr, input_ptr, window_size_in_bytes);
}
} // namespace

void CpuReshapeKernel::configure(const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst));
    ARM_COMPUTE_UNUSED(src);

    _reshape_tensor_fn = reshape_tensor_per_element_selector;
    // Configure kernel window
    Window win = calculate_max_window(*dst);

    ICpuKernel::configure(win);
}

Status CpuReshapeKernel::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst));
    return Status{};
}

void CpuReshapeKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    const auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto       dst = tensors.get_tensor(TensorType::ACL_DST);
    _reshape_tensor_fn(window, src, dst);
}

const char *CpuReshapeKernel::name() const
{
    return "CpuReshapeKernel";
}

size_t CpuReshapeKernel::get_mws(const CPUInfo &platform, size_t thread_count) const
{
    ARM_COMPUTE_UNUSED(thread_count);
    ARM_COMPUTE_UNUSED(platform);

    return ICPPKernel::default_mws;
}

void CpuReshapeKernel::prepare(ITensorPack &tensors)
{
    const auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto       dst = tensors.get_tensor(TensorType::ACL_DST);

    const ITensorInfo *src_info = src->info();
    const ITensorInfo *dst_info = dst->info();

    // Calculate kernel window based on the padding info
    Window win;

    const bool src_has_holes      = has_holes(*src_info, src_info->num_dimensions() - 1);
    const bool dst_has_holes      = has_holes(*dst_info, dst_info->num_dimensions() - 1);
    const bool src_has_holes_in_x = has_holes(*src_info, Window::DimX);
    const bool dst_has_holes_in_x = has_holes(*dst_info, Window::DimX);
    const auto src_row_size       = static_cast<int>(src_info->tensor_shape()[0]);
    const auto dst_row_size       = static_cast<int>(dst_info->tensor_shape()[0]);

    if (!src_has_holes && !dst_has_holes)
    {
        std::tie(win, _split_dimension) = calculate_squashed_or_max_window(*dst_info);
        /*
            Copy the tensor per window. If the src and dst tensors
            are contiguous memory allocations without any holes or
            padding, then the tensor is squashed to 1D window and
            we can use use a single memcopy call to copy the whole
            window in reshape_tensor_per_window fn
        */
        _reshape_tensor_fn = reshape_tensor_per_window;
    }
    else
    {
        win = calculate_max_window(*dst_info);
        /*
            Copy tensor row by row if src and dst have no holes in X
            dim and they have the same number of elements in their rows
        */
        if (!src_has_holes_in_x && !dst_has_holes_in_x && (src_row_size == dst_row_size))
        {
            _reshape_tensor_fn = reshape_tensor_per_row;
        }
        else
        {
            /*
                Fall back to the element wise copy
            */
            _reshape_tensor_fn = reshape_tensor_per_element_selector;
        }
    }

    ICPPKernel::configure(win);
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
/** [NEReshapeLayerKernel Kernel] **/
