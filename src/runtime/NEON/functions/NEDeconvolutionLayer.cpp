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
#include "arm_compute/runtime/NEON/functions/NEDeconvolutionLayer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"

using namespace arm_compute;

NEDeconvolutionLayer::NEDeconvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager) // NOLINT
    : _memory_group(std::move(memory_manager)),
      _scale_f(),
      _conv_f(),
      _scaled_output()
{
}

void NEDeconvolutionLayer::configure(ITensor *input, const ITensor *weights, const ITensor *bias, ITensor *output, const PadStrideInfo &info,
                                     unsigned int ax, unsigned int ay, float upscalex, float upscaley)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON(weights->info()->dimension(0) != weights->info()->dimension(1));
    ARM_COMPUTE_ERROR_ON(weights->info()->dimension(0) < 1);

    auto out_dims = deconvolution_output_dimensions(input->info()->dimension(0), input->info()->dimension(1), weights->info()->dimension(0), weights->info()->dimension(1),
                                                    info.pad().first, info.pad().second, ax, ay, upscalex, upscaley, info.round());

    const TensorShape output_shape = deconvolution_output_shape(out_dims, input->info()->tensor_shape(), weights->info()->tensor_shape());

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->fixed_point_position());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output, weights, bias);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output, weights, bias);

    ARM_COMPUTE_ERROR_ON_MSG(output->info()->dimension(Window::DimX) != output_shape.x(), "Output's width is invalid.");
    ARM_COMPUTE_ERROR_ON_MSG(output->info()->dimension(Window::DimY) != output_shape.y(), "Output's height is invalid.");
    ARM_COMPUTE_ERROR_ON_MSG(output->info()->dimension(Window::DimZ) != output_shape.z(), "Output's depth is invalid.");

    _memory_group.manage(&_scaled_output);

    // configure scale function
    //Init and allocate intermmidiate tensor for output, same size as input but the first two axis are the same as the output tensor
    TensorShape scale_out_shape(input->info()->tensor_shape());
    scale_out_shape.set(0, output->info()->dimension(0));
    scale_out_shape.set(1, output->info()->dimension(1));
    TensorInfo scale_out_info(scale_out_shape, 1, input->info()->data_type(), input->info()->fixed_point_position());
    _scaled_output.allocator()->init(scale_out_info);
    const unsigned int kernel_size = weights->info()->dimension(0);
    // Padding for the upsampled image is calculated with the equiation: p' = k - p - 1, where k is kernel size and p is the input padding
    ARM_COMPUTE_ERROR_ON(info.pad().first > (kernel_size - 1));
    const unsigned int  tr_px     = kernel_size - info.pad().first - 1;
    const unsigned int  tr_py     = kernel_size - info.pad().second - 1;
    const unsigned int  tr_stride = 1;
    const PadStrideInfo transposed_info(tr_stride, tr_stride, tr_px, tr_py);
    _scale_f.configure(input, &_scaled_output, std::make_pair(ax, ay), std::make_pair(info.stride().first - 1u, info.stride().second - 1u), transposed_info);
    // setup the function to convolve the upscaled output
    switch(kernel_size)
    {
        case 1:
        {
            _conv_f.configure(&_scaled_output, weights, bias, output, PadStrideInfo(1, 1, 0, 0, DimensionRoundingType::CEIL));
            break;
        }
        case 3:
        {
            _conv_f.configure(&_scaled_output, weights, bias, output, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL));
            break;
        }
        case 5:
        {
            _conv_f.configure(&_scaled_output, weights, bias, output, PadStrideInfo(1, 1, 2, 2, DimensionRoundingType::CEIL));
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Not supported");
            break;
        }
    }
    _scaled_output.allocator()->allocate();
}

void NEDeconvolutionLayer::run()
{
    _memory_group.acquire();
    _scale_f.run();
    _conv_f.run();
    _memory_group.release();
}
