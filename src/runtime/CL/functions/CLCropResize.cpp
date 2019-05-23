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

#include "arm_compute/core/CL/CLHelpers.h"

#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/functions/CLCropResize.h"

#include <cstddef>

namespace arm_compute
{
namespace
{
inline void configure_crop(const ICLTensor *input, ICLTensor *crop_boxes, ICLTensor *box_ind, ICLTensor *output, uint32_t crop_box_ind, Coordinates &start, Coordinates &end, uint32_t &batch_index)
{
    batch_index = *(reinterpret_cast<int32_t *>(box_ind->ptr_to_element(Coordinates(crop_box_ind))));

    // _crop_box_ind is used to index crop_boxes and retrieve the appropriate crop box.
    // The crop box is specified by normalized coordinates [y0, x0, y1, x1].
    const float x0 = *reinterpret_cast<const float *>(crop_boxes->ptr_to_element(Coordinates(1, crop_box_ind)));
    const float y0 = *reinterpret_cast<const float *>(crop_boxes->ptr_to_element(Coordinates(0, crop_box_ind)));
    const float x1 = *reinterpret_cast<const float *>(crop_boxes->ptr_to_element(Coordinates(3, crop_box_ind)));
    const float y1 = *reinterpret_cast<const float *>(crop_boxes->ptr_to_element(Coordinates(2, crop_box_ind)));
    // The normalized coordinates are scaled to retrieve the floating point image coordinates which are rounded to integers.
    start = Coordinates(std::floor(x0 * (input->info()->tensor_shape()[1] - 1) + 0.5f),
                        std::floor(y0 * (input->info()->tensor_shape()[2] - 1) + 0.5f));
    end = Coordinates(std::floor(x1 * (input->info()->tensor_shape()[1] - 1) + 0.5f),
                      std::floor(y1 * (input->info()->tensor_shape()[2] - 1) + 0.5f));
    const TensorShape out_shape(input->info()->tensor_shape()[0], abs(end[0] - start[0]) + 1, abs(end[1] - start[1]) + 1);
    output->info()->set_tensor_shape(out_shape);
}

inline void run_crop(const ICLTensor *input, ICLTensor *output, uint32_t batch_index, Coordinates start, Coordinates end, float extrapolation_value)
{
    bool is_width_flipped  = end[0] < start[0];
    bool is_height_flipped = end[1] < start[1];
    /** The number of rows out of bounds at the start and end of output. */
    std::array<int32_t, 2> rows_out_of_bounds{ 0 };
    /** The number of columns out of bounds at the start and end of output. */
    std::array<int32_t, 2> cols_out_of_bounds{ 0 };
    if(is_height_flipped)
    {
        rows_out_of_bounds[0] = start[1] >= static_cast<int32_t>(input->info()->dimension(2)) ? std::min(start[1] - input->info()->dimension(2) + 1, output->info()->dimension(2)) : 0;
        rows_out_of_bounds[1] = end[1] < 0 ? std::min(-end[1], static_cast<int32_t>(output->info()->dimension(2))) : 0;
    }
    else
    {
        rows_out_of_bounds[0] = start[1] < 0 ? std::min(-start[1], static_cast<int32_t>(output->info()->dimension(2))) : 0;
        rows_out_of_bounds[1] = end[1] >= static_cast<int32_t>(input->info()->dimension(2)) ? std::min(end[1] - input->info()->dimension(2) + 1, output->info()->dimension(2)) : 0;
    }
    if(is_width_flipped)
    {
        cols_out_of_bounds[0] = start[0] >= static_cast<int32_t>(input->info()->dimension(1)) ? std::min(start[0] - input->info()->dimension(1) + 1, output->info()->dimension(1)) : 0;
        cols_out_of_bounds[1] = end[0] < 0 ? std::min(-end[0], static_cast<int32_t>(output->info()->dimension(1))) : 0;
    }
    else
    {
        cols_out_of_bounds[0] = start[0] < 0 ? std::min(-start[0], static_cast<int32_t>(output->info()->dimension(1))) : 0;
        cols_out_of_bounds[1] = end[0] >= static_cast<int32_t>(input->info()->dimension(1)) ? std::min(end[0] - input->info()->dimension(1) + 1, output->info()->dimension(1)) : 0;
    }

    Window full_window = calculate_max_window(*output->info());

    //  Full output window:
    //  --------------------------------
    //  |          Out of bounds       |
    //  |          rows before         |
    //  |------------------------------|
    //  | Out of | In         | Out of |
    //  | bounds | bounds     | bounds |
    //  | cols   | elements   | cols   |
    //  | before | copied     | after  |
    //  |        | from input |        |
    //  |------------------------------|
    //  |        Out of bounds         |
    //  |        rows after            |
    //  |------------------------------|
    // Use a separate output window for each section of the full output window.
    // Fill all output rows that have no elements that are within the input bounds
    // with the extrapolation value using memset.
    // First for the rows before the in bounds rows.
    if(rows_out_of_bounds[0] > 0)
    {
        Window slice_fill_rows_before(full_window);
        slice_fill_rows_before.set(2, Window::Dimension(0, rows_out_of_bounds[0], 1));
        auto kernel = arm_compute::support::cpp14::make_unique<CLMemsetKernel>();
        kernel->configure(output, extrapolation_value, &slice_fill_rows_before);
        CLScheduler::get().enqueue(*kernel);
    }

    Window slice_in(full_window);
    slice_in.set(2, Window::Dimension(rows_out_of_bounds[0], output->info()->dimension(2) - rows_out_of_bounds[1], 1));
    slice_in.set(1, Window::Dimension(cols_out_of_bounds[0], output->info()->dimension(1) - cols_out_of_bounds[1], 1));

    int rows_in_bounds = static_cast<int32_t>(output->info()->dimension(2)) - rows_out_of_bounds[0] - rows_out_of_bounds[1];
    if(rows_in_bounds > 0)
    {
        // Fill all elements that share a row with an in bounds element with the extrapolation value.
        if(cols_out_of_bounds[0] > 0)
        {
            Window slice_fill_cols_before(slice_in);
            slice_fill_cols_before.set(1, Window::Dimension(0, cols_out_of_bounds[0], 1));
            auto kernel = arm_compute::support::cpp14::make_unique<CLMemsetKernel>();
            kernel->configure(output, extrapolation_value, &slice_fill_cols_before);
            CLScheduler::get().enqueue(*kernel);
        }

        if(cols_out_of_bounds[1] > 0)
        {
            Window slice_fill_cols_after(slice_in);
            slice_fill_cols_after.set(1, Window::Dimension(output->info()->dimension(1) - cols_out_of_bounds[1], output->info()->dimension(1), 1));
            auto kernel = arm_compute::support::cpp14::make_unique<CLMemsetKernel>();
            kernel->configure(output, extrapolation_value, &slice_fill_cols_after);
            CLScheduler::get().enqueue(*kernel);
        }

        // Copy all elements within the input bounds from the input tensor.
        int cols_in_bounds = static_cast<int32_t>(output->info()->dimension(1)) - cols_out_of_bounds[0] - cols_out_of_bounds[1];
        if(cols_in_bounds > 0)
        {
            Coordinates2D start_in{ is_width_flipped ? start[0] - cols_out_of_bounds[0] : start[0] + cols_out_of_bounds[0],
                                    is_height_flipped ? start[1] - rows_out_of_bounds[0] : start[1] + rows_out_of_bounds[0] };
            Coordinates2D end_in{ is_width_flipped ? start_in.x - cols_in_bounds + 1 : start_in.x + cols_in_bounds - 1,
                                  is_height_flipped ? start_in.y - rows_in_bounds + 1 : start_in.y + rows_in_bounds - 1 };
            auto kernel = arm_compute::support::cpp14::make_unique<CLCropKernel>();

            kernel->configure(input, output, start_in, end_in, batch_index, extrapolation_value, &slice_in);
            CLScheduler::get().enqueue(*kernel);
        }
    }

    // Fill all rows after the in bounds elements with the extrapolation value.
    if(rows_out_of_bounds[1] > 0)
    {
        Window slice_fill_rows_after(full_window);
        slice_fill_rows_after.set(2, Window::Dimension(output->info()->dimension(2) - rows_out_of_bounds[1], output->info()->dimension(2), 1));
        auto kernel = arm_compute::support::cpp14::make_unique<CLMemsetKernel>();
        kernel->configure(output, extrapolation_value, &slice_fill_rows_after);
        CLScheduler::get().enqueue(*kernel);
    }
}
} // namespace

CLCropResize::CLCropResize()
    : _input(nullptr), _boxes(nullptr), _box_ind(nullptr), _output(nullptr), _num_boxes(0), _method(), _extrapolation_value(0), _scale(), _copy(), _crop_results(), _scaled_results()
{
}

Status CLCropResize::validate(const ITensorInfo *input, ITensorInfo *boxes, ITensorInfo *box_ind, const ITensorInfo *output,
                              Coordinates2D crop_size, InterpolationPolicy method, float extrapolation_value)
{
    ARM_COMPUTE_RETURN_ERROR_ON(crop_size.x <= 0 || crop_size.y <= 0);
    ARM_COMPUTE_RETURN_ERROR_ON(method == InterpolationPolicy::AREA);
    ARM_COMPUTE_RETURN_ERROR_ON(boxes->tensor_shape()[0] != 4);
    ARM_COMPUTE_RETURN_ERROR_ON(boxes->tensor_shape()[1] != box_ind->tensor_shape()[0]);
    TensorInfo temp_info;
    ARM_COMPUTE_RETURN_ON_ERROR(CLCropKernel::validate(input->clone().get(), &temp_info, { 0, 0 }, { 1, 1 }, input->dimension(3) - 1, extrapolation_value));
    if(output->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(output, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
        TensorShape out_shape(input->tensor_shape()[0], crop_size.x, crop_size.y, boxes->tensor_shape()[1]);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), out_shape);
    }
    return Status{};
}

void CLCropResize::configure(const ICLTensor *input, ICLTensor *boxes, ICLTensor *box_ind, ICLTensor *output, Coordinates2D crop_size,
                             InterpolationPolicy method, float extrapolation_value)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLCropResize::validate(input->info(), boxes->info(), box_ind->info(), output->info(), crop_size, method, extrapolation_value));

    _num_boxes = boxes->info()->tensor_shape()[1];
    TensorShape out_shape(input->info()->tensor_shape()[0], crop_size.x, crop_size.y);

    _input               = input;
    _boxes               = boxes;
    _box_ind             = box_ind;
    _output              = output;
    _method              = method;
    _extrapolation_value = extrapolation_value;

    // For each crop box:
    // - The initial cropped image is produced as specified by boxes[i] from the 3D image input[box_ind[i]].
    //   Possibly using a CLCropKernel and up to four CLMemsetKernels.
    // - A tensor is required to hold this initial cropped image.
    // - A scale function is used to resize the cropped image to the size specified by crop_size.
    // - A tensor is required to hold the final scaled image before it is copied into the 4D output
    //   that will hold all final cropped and scaled 3D images using CLCopyKernel.
    for(unsigned int i = 0; i < _num_boxes; ++i)
    {
        auto       crop_tensor = support::cpp14::make_unique<CLTensor>();
        TensorInfo crop_result_info(1, DataType::F32);
        crop_result_info.set_data_layout(DataLayout::NHWC);
        crop_tensor->allocator()->init(crop_result_info);
        _crop_results.emplace_back(std::move(crop_tensor));

        auto       scale_tensor = support::cpp14::make_unique<CLTensor>();
        TensorInfo scaled_result_info(out_shape, 1, DataType::F32);
        scaled_result_info.set_data_layout(DataLayout::NHWC);
        scale_tensor->allocator()->init(scaled_result_info);
        _scaled_results.emplace_back(std::move(scale_tensor));
    }
}

void CLCropResize::run()
{
    ARM_COMPUTE_ERROR_ON_MSG(_output == nullptr, "Unconfigured function");
    // The contents of _boxes and _box_ind are required to calculate the shape
    // of the initial cropped image and thus are required to configure the
    // kernels used for cropping and scaling.
    _boxes->map(CLScheduler::get().queue());
    _box_ind->map(CLScheduler::get().queue());
    for(unsigned int i = 0; i < _num_boxes; ++i)
    {
        // Size of the crop box in _boxes and thus the shape of _crop_results[i]
        // may not be known until run-time and so the kernels cannot be configured until then.
        uint32_t    batch_index;
        Coordinates start{};
        Coordinates end{};
        configure_crop(_input, _boxes, _box_ind, _crop_results[i].get(), i, start, end, batch_index);

        auto scale_kernel = support::cpp14::make_unique<CLScale>();
        scale_kernel->configure(_crop_results[i].get(), _scaled_results[i].get(), _method, BorderMode::CONSTANT, PixelValue(_extrapolation_value), SamplingPolicy::TOP_LEFT);
        _scale.emplace_back(std::move(scale_kernel));

        Window win = calculate_max_window(*_output->info());
        win.set(3, Window::Dimension(i, i + 1, 1));

        auto copy_kernel = support::cpp14::make_unique<CLCopyKernel>();
        copy_kernel->configure(_scaled_results[i].get(), _output, PaddingList(), &win);
        _copy.emplace_back(std::move(copy_kernel));

        _crop_results[i]->allocator()->allocate();
        _scaled_results[i]->allocator()->allocate();

        run_crop(_input, _crop_results[i].get(), batch_index, start, end, _extrapolation_value);
    }
    _boxes->unmap(CLScheduler::get().queue());
    _box_ind->unmap(CLScheduler::get().queue());
    CLScheduler::get().sync();
    for(auto &kernel : _scale)
    {
        kernel->run();
    }
    CLScheduler::get().sync();
    for(auto &kernel : _copy)
    {
        CLScheduler::get().enqueue(*kernel, true);
    }
    CLScheduler::get().sync();
}
} // namespace arm_compute