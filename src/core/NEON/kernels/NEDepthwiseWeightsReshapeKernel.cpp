/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEDepthwiseWeightsReshapeKernel.h"

#include "arm_compute/core/AccessWindowTranspose.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

using namespace arm_compute;

NEDepthwiseWeightsReshapeKernel::NEDepthwiseWeightsReshapeKernel()
    : _input(nullptr), _output(nullptr), _biases(nullptr)
{
}

void NEDepthwiseWeightsReshapeKernel::configure(const ITensor *input, ITensor *output, const ITensor *biases)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(2) != output->info()->dimension(1));
    ARM_COMPUTE_ERROR_ON(output->info()->dimension(0) != (input->info()->dimension(0) * input->info()->dimension(1) + ((biases != nullptr) ? 1 : 0)));

    if(biases != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, biases);
        ARM_COMPUTE_ERROR_ON(biases->info()->dimension(0) != input->info()->dimension(2));
        ARM_COMPUTE_ERROR_ON(biases->info()->num_dimensions() > 1);
    }

    _input  = input;
    _output = output;
    _biases = biases;

    // Configure  kernel window
    Window win = calculate_max_window(*input->info(), Steps());
    // The NEDepthwiseWeightsReshapeKernel doesn't need padding so update_window_and_padding() can be skipped
    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NEDepthwiseWeightsReshapeKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);

    const int input_w         = _input->info()->dimension(0);
    const int output_stride_x = _output->info()->strides_in_bytes().x();
    const int output_stride_y = _output->info()->strides_in_bytes().y();

    Window window_in(window);
    // The first three dimensions of the input are increased by the inner loops
    window_in.set(Window::DimX, Window::Dimension(0, _input->info()->dimension(0), _input->info()->dimension(0)));
    window_in.set(Window::DimY, Window::Dimension(0, _input->info()->dimension(1), 1));
    window_in.set(Window::DimZ, Window::Dimension(0, _input->info()->dimension(2), 1));

    // Setup output window
    Window window_out;
    window_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    window_out.set(Window::DimY, Window::Dimension(0, 0, 0));

    Iterator in(_input, window_in);
    Iterator out(_output, window_out);

    execute_window_loop(window_in, [&](const Coordinates & id)
    {
        auto input_ptr  = reinterpret_cast<float *>(in.ptr());
        auto output_ptr = reinterpret_cast<float *>(out.ptr() + id.y() * input_w * output_stride_x + id.z() * output_stride_y);

        for(int i = 0; i < input_w; ++i, ++input_ptr)
        {
            *(output_ptr + i) = *input_ptr;
        }

        if(_biases != nullptr)
        {
            *(output_ptr + input_w) = *(reinterpret_cast<float *>(_biases->ptr_to_element(Coordinates(id.z()))));
        }
    },
    in, out);
}
