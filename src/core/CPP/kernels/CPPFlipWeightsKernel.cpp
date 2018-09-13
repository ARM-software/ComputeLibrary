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
#include "arm_compute/core/CPP/kernels/CPPFlipWeightsKernel.h"

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

CPPFlipWeightsKernel::CPPFlipWeightsKernel()
    : _input(nullptr), _output(nullptr), _func(nullptr)
{
}

template <typename T>
void CPPFlipWeightsKernel::flip_weights(const Window &window_input)
{
    // Create iterators
    Iterator in(_input, window_input);

    const DataLayout data_layout = _input->info()->data_layout();
    const size_t     idx_w       = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const size_t     idx_h       = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    const int kernel_width  = _input->info()->dimension(idx_w);
    const int kernel_height = _input->info()->dimension(idx_h);

    execute_window_loop(window_input, [&](const Coordinates & id)
    {
        const unsigned int x = kernel_width - id[idx_w] - 1;
        const unsigned int y = kernel_height - id[idx_h] - 1;
        Coordinates        output_coord(id);
        output_coord.set(idx_w, x);
        output_coord.set(idx_h, y);
        *(reinterpret_cast<T *>(_output->ptr_to_element(output_coord))) = *(reinterpret_cast<const T *>(in.ptr()));
    },
    in);
}

void CPPFlipWeightsKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    _input  = input;
    _output = output;

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps());

    // The CPPFlipWeightsKernel doesn't need padding so update_window_and_padding() can be skipped
    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

    ICPPKernel::configure(win);

    switch(input->info()->data_type())
    {
        case DataType::F32:
            _func = &CPPFlipWeightsKernel::flip_weights<float>;
            break;
        case DataType::F16:
            _func = &CPPFlipWeightsKernel::flip_weights<half>;
            break;
        case DataType::QASYMM8:
            _func = &CPPFlipWeightsKernel::flip_weights<uint8_t>;
            break;
        default:
            ARM_COMPUTE_ERROR("Not supported");
    }
}

void CPPFlipWeightsKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICPPKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}
