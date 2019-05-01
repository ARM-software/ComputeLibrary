/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "arm_compute/runtime/NEON/functions/NECropResize.h"

#include <cstddef>

namespace arm_compute
{
NECropResize::NECropResize()
    : _output(nullptr), _num_boxes(0), _method(), _extrapolation_value(0), _crop(), _scale(), _crop_results(), _scaled_results()
{
}

Status NECropResize::validate(const ITensorInfo *input, const ITensorInfo *boxes, const ITensorInfo *box_ind, const ITensorInfo *output,
                              Coordinates2D crop_size, InterpolationPolicy method, float extrapolation_value)
{
    ARM_COMPUTE_RETURN_ERROR_ON(crop_size.x <= 0 || crop_size.y <= 0);
    ARM_COMPUTE_RETURN_ERROR_ON(method == InterpolationPolicy::AREA);
    TensorInfo temp_info;
    ARM_COMPUTE_RETURN_ON_ERROR(NECropKernel::validate(input->clone().get(), boxes->clone().get(), box_ind->clone().get(), &temp_info, boxes->tensor_shape()[1] - 1, extrapolation_value));
    if(output->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(output, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
        TensorShape out_shape(input->tensor_shape()[0], crop_size.x, crop_size.y, boxes->tensor_shape()[1]);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), out_shape);
    }
    return Status{};
}

void NECropResize::configure(const ITensor *input, const ITensor *boxes, const ITensor *box_ind, ITensor *output, Coordinates2D crop_size,
                             InterpolationPolicy method, float extrapolation_value)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(NECropResize::validate(input->info(), boxes->info(), box_ind->info(), output->info(), crop_size, method, extrapolation_value));

    _num_boxes = boxes->info()->tensor_shape()[1];
    TensorShape out_shape(input->info()->tensor_shape()[0], crop_size.x, crop_size.y);

    _output              = output;
    _method              = method;
    _extrapolation_value = extrapolation_value;

    // For each crop box:
    // - A crop kernel is used to extract the initial cropped image as specified by boxes[i] from the 3D image input[box_ind[i]].
    // - A tensor is required to hold this initial cropped image.
    // - A scale function is used to resize the cropped image to the size specified by crop_size.
    // - A tensor is required to hold the final scaled image before it is copied into the 4D output
    //   that will hold all final cropped and scaled 3D images.
    _crop.reserve(_num_boxes);
    _crop_results.reserve(_num_boxes);
    _scaled_results.reserve(_num_boxes);
    _scale.reserve(_num_boxes);

    for(unsigned int i = 0; i < _num_boxes; ++i)
    {
        auto       crop_tensor = support::cpp14::make_unique<Tensor>();
        TensorInfo crop_result_info(1, DataType::F32);
        crop_result_info.set_data_layout(DataLayout::NHWC);
        crop_tensor->allocator()->init(crop_result_info);

        auto       scale_tensor = support::cpp14::make_unique<Tensor>();
        TensorInfo scaled_result_info(out_shape, 1, DataType::F32);
        scaled_result_info.set_data_layout(DataLayout::NHWC);
        scale_tensor->allocator()->init(scaled_result_info);

        auto crop_kernel  = support::cpp14::make_unique<NECropKernel>();
        auto scale_kernel = support::cpp14::make_unique<NEScale>();
        crop_kernel->configure(input, boxes, box_ind, crop_tensor.get(), i, _extrapolation_value);

        _crop.emplace_back(std::move(crop_kernel));
        _scaled_results.emplace_back(std::move(scale_tensor));
        _crop_results.emplace_back(std::move(crop_tensor));
        _scale.emplace_back(std::move(scale_kernel));
    }
}

void NECropResize::run()
{
    ARM_COMPUTE_ERROR_ON_MSG(_output == nullptr, "Unconfigured function");

    for(unsigned int i = 0; i < _num_boxes; ++i)
    {
        // Size of the crop box in _boxes and thus the shape of _crop_results[i]
        // may not be known until run-time and so the kernels cannot be configured until then.
        _crop[i]->configure_output_shape();
        _crop_results[i]->allocator()->allocate();
        NEScheduler::get().schedule(_crop[i].get(), Window::DimZ);

        // Scale the cropped image.
        _scale[i]->configure(_crop_results[i].get(), _scaled_results[i].get(), _method, BorderMode::CONSTANT, PixelValue(_extrapolation_value), SamplingPolicy::TOP_LEFT, false);
        _scaled_results[i]->allocator()->allocate();
        _scale[i]->run();

        // Copy scaled image into output.
        std::copy_n(_scaled_results[i]->buffer(), _scaled_results[i]->info()->total_size(), _output->ptr_to_element(Coordinates(0, 0, 0, i)));
    }
}
} // namespace arm_compute