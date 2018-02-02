/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEDeconvolutionLayer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

NEDeconvolutionLayer::NEDeconvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager) // NOLINT
    : _memory_group(std::move(memory_manager)),
      _conv_f(),
      _scaled_output(),
      _input(nullptr),
      _info(),
      _inner_border()
{
}

void NEDeconvolutionLayer::configure(ITensor *input, const ITensor *weights, const ITensor *bias, ITensor *output, const PadStrideInfo &info,
                                     unsigned int inner_border_right, unsigned int inner_border_top)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON(weights->info()->dimension(0) != weights->info()->dimension(1));
    ARM_COMPUTE_ERROR_ON(weights->info()->dimension(0) != 1 && weights->info()->dimension(0) != 3 && weights->info()->dimension(0) != 5);

    _input        = input;
    _info         = info;
    _inner_border = std::make_pair(inner_border_right, inner_border_top);

    const unsigned int stride_x = info.stride().first;
    const unsigned int stride_y = info.stride().second;
    auto               out_dims = deconvolution_output_dimensions(input->info()->dimension(0), input->info()->dimension(1), weights->info()->dimension(0), weights->info()->dimension(1),
                                                                  info.pad().first, info.pad().second, inner_border_right, inner_border_top, stride_x, stride_y);

    const TensorShape output_shape = deconvolution_output_shape(out_dims, input->info()->tensor_shape(), weights->info()->tensor_shape());

    ARM_COMPUTE_UNUSED(output_shape);
    ARM_COMPUTE_ERROR_ON_MSG(output->info()->dimension(Window::DimX) != output_shape.x(), "Output's width is invalid.");
    ARM_COMPUTE_ERROR_ON_MSG(output->info()->dimension(Window::DimY) != output_shape.y(), "Output's height is invalid.");
    ARM_COMPUTE_ERROR_ON_MSG(output->info()->dimension(Window::DimZ) != output_shape.z(), "Output's depth is invalid.");

    _memory_group.manage(&_scaled_output);

    // configure scale function
    // Init and allocate intermmidiate tensor for output, same size as input but the first two axis are the same as the output tensor
    const TensorInfo scale_out_info(compute_deconvolution_shape(*input->info(), stride_x, stride_y, inner_border_right, inner_border_top, info), 1, input->info()->data_type(),
                                    input->info()->fixed_point_position());
    _scaled_output.allocator()->init(scale_out_info);

    // setup the function to convolve the upscaled output
    const PadStrideInfo conv_info(1, 1, 0, 0, 0, 0, DimensionRoundingType::CEIL);
    _conv_f.configure(&_scaled_output, weights, bias, output, conv_info);
    _scaled_output.allocator()->allocate();
}

void NEDeconvolutionLayer::run()
{
    _memory_group.acquire();

    // Initialize _scaled_output buffer
    const int width_in      = _input->info()->dimension(0);
    const int height_in     = _input->info()->dimension(1);
    const int width_scaled  = _scaled_output.info()->dimension(0);
    const int height_scaled = _scaled_output.info()->dimension(1);
    const int num_2d_slices = _input->info()->tensor_shape().total_size() / (width_in * height_in);
    const int stride_x      = _info.stride().first;
    const int stride_y      = _info.stride().second;

    std::fill_n(_scaled_output.buffer(), _scaled_output.info()->total_size(), 0);

    // scaled_output is the input for the forward convolution. We copy the input elements to scaled_output
    // and insert rows and columns with zeroes depending on the stride values.
    for(int slice = 0; slice < num_2d_slices; ++slice)
    {
        const int start_x = _info.pad().first;
        const int start_y = _inner_border.second + _info.pad().second;
        const int end_y   = height_scaled - _info.pad().second;
        const int end_x   = width_scaled - _inner_border.first - _info.pad().first;

        for(int yi = start_y, in_y = 0; yi < end_y; yi += stride_y, in_y++)
        {
            for(int xi = start_x, in_x = 0; xi < end_x; xi += stride_x, in_x++)
            {
                const auto in = *(reinterpret_cast<float *>(_input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(in_x, in_y, slice))));
                *(reinterpret_cast<float *>(_scaled_output.buffer() + _scaled_output.info()->offset_element_in_bytes(Coordinates(xi, yi, slice)))) = in;
            }
        }
    }

    _conv_f.run();
    _memory_group.release();
}
