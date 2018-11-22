/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEPermuteKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

namespace
{
#include "arm_compute/core/NEON/kernels/convolution/common/shims.hpp"
} // namespace

#include <cstddef>
#include <cstdint>

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const PermutationVector &perm)
{
    //Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use NEON FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S8, DataType::QASYMM8,
                                                         DataType::U16, DataType::S16,
                                                         DataType::U32, DataType::S32,
                                                         DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((perm != PermutationVector{ 2U, 0U, 1U })
                                    && (perm != PermutationVector{ 1U, 2U, 0U }),
                                    "Only [2, 0, 1] and [1, 2, 0] permutation is supported");

    const TensorShape output_shape = misc::shape_calculator::compute_permutation_output_shape(*input, perm);

    // Validate configured output
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}
} // namespace

template <typename T>
void NEPermuteKernel::run_permute(const Window &window)
{
    // Input window
    Window window_in = window;
    window_in.set(Window::DimX, Window::Dimension(window.x().start(), window.x().end(), window.x().end() - window.x().start()));
    window_in.set(Window::DimY, Window::Dimension(window.y().start(), window.y().end(), window.y().end() - window.y().start()));
    window_in.set(Window::DimZ, Window::Dimension(window.z().start(), window.z().end(), window.z().end() - window.z().start()));
    window_in.set(3, Window::Dimension(window[3].start(), window[3].end(), window[3].end() - window[3].start()));

    // Output window
    Window                  window_out(window);
    const Window::Dimension zero_window = Window::Dimension(0, 0, 0);
    for(size_t d = 0; d <= _perm.num_dimensions(); ++d)
    {
        window_out.set(d, zero_window);
    }

    // Create iterators
    Iterator in(_input, window_in);
    Iterator out(_output, window_out);

    // CHW -> HWC
    if(_perm == PermutationVector{ 2U, 0U, 1U })
    {
        const int in_row_stride     = _input->info()->strides_in_bytes().y() / sizeof(T);
        const int in_channel_stride = _input->info()->strides_in_bytes().z() / sizeof(T);
        const int in_batch_stride   = _input->info()->strides_in_bytes()[3] / sizeof(T);

        const int out_channel_stride = _output->info()->strides_in_bytes().x() / sizeof(T);
        const int out_col_stride     = _output->info()->strides_in_bytes().y() / sizeof(T);
        const int out_row_stride     = _output->info()->strides_in_bytes().z() / sizeof(T);
        const int out_batch_stride   = _output->info()->strides_in_bytes()[3] / sizeof(T);

        const int n_cols     = _input->info()->tensor_shape().x();
        const int n_rows     = window_in.y().step();
        const int n_channels = _input->info()->tensor_shape().z();
        const int n_batches  = _input->info()->tensor_shape()[3];

        execute_window_loop(window_in, [&](const Coordinates & id)
        {
            const int idx = id[0] * out_col_stride + id[1] * out_row_stride + id[2] * out_channel_stride;
            reorder::nchw_to_nhwc(reinterpret_cast<const T *>(in.ptr()), reinterpret_cast<T *>(out.ptr()) + idx,
                                  n_batches, n_channels, n_rows, n_cols,
                                  in_batch_stride, in_channel_stride, in_row_stride,
                                  out_batch_stride, out_row_stride, out_col_stride);
        },
        in, out);
    }
    // HWC -> CHW
    else if(_perm == PermutationVector{ 1U, 2U, 0U })
    {
        const int in_col_stride   = _input->info()->strides_in_bytes().y() / sizeof(T);
        const int in_row_stride   = _input->info()->strides_in_bytes().z() / sizeof(T);
        const int in_batch_stride = _input->info()->strides_in_bytes()[3] / sizeof(T);

        const int out_col_stride     = _output->info()->strides_in_bytes().x() / sizeof(T);
        const int out_row_stride     = _output->info()->strides_in_bytes().y() / sizeof(T);
        const int out_channel_stride = _output->info()->strides_in_bytes().z() / sizeof(T);
        const int out_batch_stride   = _output->info()->strides_in_bytes()[3] / sizeof(T);

        const int n_channels = _input->info()->tensor_shape().x();
        const int n_cols     = window_in.y().step();
        const int n_rows     = _input->info()->tensor_shape().z();
        const int n_batches  = _input->info()->tensor_shape()[3];

        execute_window_loop(window_in, [&](const Coordinates & id)
        {
            const int idx = id[0] * out_channel_stride + id[1] * out_col_stride + id[2] * out_row_stride;
            reorder::nhwc_to_nchw(reinterpret_cast<const T *>(in.ptr()), reinterpret_cast<T *>(out.ptr()) + idx,
                                  n_batches, n_rows, n_cols, n_channels,
                                  in_batch_stride, in_row_stride, in_col_stride,
                                  out_batch_stride, out_channel_stride, out_row_stride);
        },
        in, out);
    }
    else
    {
        ARM_COMPUTE_ERROR("Unsupported permutation vector");
    }
}

NEPermuteKernel::NEPermuteKernel()
    : _func(), _input(nullptr), _output(nullptr), _perm()
{
}

void NEPermuteKernel::configure(const ITensor *input, ITensor *output, const PermutationVector &perm)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    const TensorShape output_shape = misc::shape_calculator::compute_permutation_output_shape(*input->info(), perm);
    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape));

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), perm));

    _input  = input;
    _output = output;
    _perm   = perm;

    switch(input->info()->element_size())
    {
        case 1:
            _func = &NEPermuteKernel::run_permute<uint8_t>;
            break;
        case 2:
            _func = &NEPermuteKernel::run_permute<uint16_t>;
            break;
        case 4:
            _func = &NEPermuteKernel::run_permute<uint32_t>;
            break;
        default:
            ARM_COMPUTE_ERROR("Element size not supported");
            break;
    }

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps());

    // The NEPermute doesn't need padding so update_window_and_padding() can be skipped
    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

    ICPPKernel::configure(win);
}

Status NEPermuteKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const PermutationVector &perm)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, perm));
    return Status{};
}

void NEPermuteKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICPPKernel::window(), window);

    if(_func != nullptr)
    {
        (this->*_func)(window);
    }
}
