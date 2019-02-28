/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/core/CPP/kernels/CPPPermuteKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <cstddef>
#include <cstdint>

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const PermutationVector &perm)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S8, DataType::QASYMM8,
                                                         DataType::U16, DataType::S16,
                                                         DataType::U32, DataType::S32,
                                                         DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(perm.num_dimensions() > 4, "Only up to 4D permutation vectors are supported");

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
void CPPPermuteKernel::run_permute(const Window &window)
{
    // Permute strides
    Strides strides      = _output->info()->strides_in_bytes();
    Strides perm_strides = strides;
    permute_strides(perm_strides, _perm);

    // Create output window
    Window                  window_out(window);
    const Window::Dimension zero_window = Window::Dimension(0, 0, 0);
    for(size_t d = 0; d <= _perm.num_dimensions(); ++d)
    {
        window_out.set(d, zero_window);
    }

    // Create iterators
    Iterator in(_input, window);
    Iterator out(_output, window_out);

    if(_input->info()->num_dimensions() <= 3)
    {
        execute_window_loop(window, [&](const Coordinates & id)
        {
            const int idx                             = id[0] * perm_strides[0] + id[1] * perm_strides[1] + id[2] * perm_strides[2];
            *(reinterpret_cast<T *>(out.ptr() + idx)) = *(reinterpret_cast<const T *>(in.ptr()));
        },
        in, out);
    }
    else if(_input->info()->num_dimensions() >= 4)
    {
        execute_window_loop(window, [&](const Coordinates & id)
        {
            const int idx                             = id[0] * perm_strides[0] + id[1] * perm_strides[1] + id[2] * perm_strides[2] + id[3] * perm_strides[3];
            *(reinterpret_cast<T *>(out.ptr() + idx)) = *(reinterpret_cast<const T *>(in.ptr()));
        },
        in, out);
    }
}

CPPPermuteKernel::CPPPermuteKernel()
    : _func(), _input(nullptr), _output(nullptr), _perm()
{
}

void CPPPermuteKernel::configure(const ITensor *input, ITensor *output, const PermutationVector &perm)
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
            _func = &CPPPermuteKernel::run_permute<uint8_t>;
            break;
        case 2:
            _func = &CPPPermuteKernel::run_permute<uint16_t>;
            break;
        case 4:
            _func = &CPPPermuteKernel::run_permute<uint32_t>;
            break;
        default:
            ARM_COMPUTE_ERROR("Element size not supported");
            break;
    }

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps());

    // The CPPPermute doesn't need padding so update_window_and_padding() can be skipped
    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

    ICPPKernel::configure(win);
}

Status CPPPermuteKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const PermutationVector &perm)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, perm));
    return Status{};
}

void CPPPermuteKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICPPKernel::window(), window);

    if(_func != nullptr)
    {
        (this->*_func)(window);
    }
}
