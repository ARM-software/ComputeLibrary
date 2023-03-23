/*
 * Copyright (c) 2019-2020, 2023 Arm Limited.
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
#include "src/core/NEON/kernels/NEBatchToSpaceLayerKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

using namespace arm_compute::misc::shape_calculator;

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *block_info, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, block_info, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(block_info, 1, DataType::S32);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);

    // Validate output if initialized
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(output->num_dimensions() > 4);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}
Status validate_arguments_static(const ITensorInfo *input, int block_shape_x, int block_shape_y, const ITensorInfo *output, const CropInfo &crop_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4);
    ARM_COMPUTE_RETURN_ERROR_ON(block_shape_x <= 0);
    ARM_COMPUTE_RETURN_ERROR_ON(block_shape_y <= 0);

    const DataLayout data_layout = input->data_layout();
    const int        idx_batch   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);
    ARM_COMPUTE_RETURN_ERROR_ON(input->tensor_shape()[idx_batch] % (block_shape_x * block_shape_y) != 0);
    // Validate output if initialized
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(output->num_dimensions() > 4);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

        const TensorShape expected_output_shape = compute_batch_to_space_shape(input->data_layout(), input->tensor_shape(), block_shape_x, block_shape_y, crop_info);
        const TensorInfo  expected_output       = output->clone()->set_tensor_shape(expected_output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &expected_output);
    }

    return Status{};
}
} // namespace

NEBatchToSpaceLayerKernel::NEBatchToSpaceLayerKernel()
    : _input(nullptr), _block_shape(nullptr), _output(nullptr), _data_layout(DataLayout::UNKNOWN), _block_shape_x(), _block_shape_y(), _crop_info()
{
}

void NEBatchToSpaceLayerKernel::configure(const ITensor *input, const ITensor *block_shape, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), block_shape->info(), output->info()));

    _input       = input;
    _block_shape = block_shape;
    _output      = output;
    _data_layout = input->info()->data_layout();

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps());
    ICPPKernel::configure(win);
}

void NEBatchToSpaceLayerKernel::configure(const ITensor *input, int32_t block_shape_x, int32_t block_shape_y, ITensor *output, const CropInfo &crop_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    const TensorShape output_shape = compute_batch_to_space_shape(input->info()->data_layout(), input->info()->tensor_shape(), block_shape_x, block_shape_y);
    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape));

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_static(input->info(), block_shape_x, block_shape_y, output->info(), crop_info));

    _input         = input;
    _output        = output;
    _block_shape_x = block_shape_x;
    _block_shape_y = block_shape_y;
    _data_layout   = input->info()->data_layout();
    _crop_info     = crop_info;

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps());
    ICPPKernel::configure(win);
}

Status NEBatchToSpaceLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *block_shape, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, block_shape, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, block_shape, output));
    return Status{};
}

Status NEBatchToSpaceLayerKernel::validate(const ITensorInfo *input, int32_t block_shape_x, int32_t block_shape_y, const ITensorInfo *output, const CropInfo &crop_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_static(input, block_shape_x, block_shape_y, output, crop_info));
    return Status{};
}

void NEBatchToSpaceLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICPPKernel::window(), window);

    if(_block_shape != nullptr)
    {
        // Retrieve the block shapes dynamically
        _block_shape_x = *(reinterpret_cast<const int *>(_block_shape->ptr_to_element(0)));
        _block_shape_y = *(reinterpret_cast<const int *>(_block_shape->ptr_to_element(1)));
    }

    const int batch_size   = _output->info()->dimension(3);
    const int element_size = _output->info()->element_size();

    Window slice_out = window.first_slice_window_3D();

    int batch_id = 0;
    // Main loop for NCHW and NHWC
    if(_data_layout == DataLayout::NCHW)
    {
        do
        {
            Iterator out(_output, slice_out);
            execute_window_loop(slice_out, [&](const Coordinates & id)
            {

                const int x = id.x();
                const int y = id.y();
                const int z = id.z();
                // Translate x, y to uncropped version
                const int x_c = x + _crop_info.left;
                const int y_c = y + _crop_info.top;

                const int   in_batch = batch_id + ((x_c % _block_shape_x) + (y_c % _block_shape_y) * _block_shape_x) * batch_size;
                const int   in_x     = x_c / _block_shape_x;
                const int   in_y     = y_c / _block_shape_y;
                Coordinates input_coords{ in_x, in_y, z, in_batch };
                memcpy(out.ptr(), _input->ptr_to_element(input_coords), element_size);
            },
            out);
            ++batch_id;
        }
        while(window.slide_window_slice_3D(slice_out));
    }
    else
    {
        // For NHWC we can perform a block copy on the Channel (first) dimension. Thus we do not need to iterate over this dimension
        slice_out.set(0U, Window::Dimension(0U, 1U, 1U));
        do
        {
            Iterator out(_output, slice_out);
            execute_window_loop(slice_out, [&](const Coordinates & id)
            {

                const int x = id.y();
                const int y = id.z();

                // Translate x, y to uncropped version
                const int x_c = x + _crop_info.left;
                const int y_c = y + _crop_info.top;

                const int   in_batch = batch_id + ((x_c % _block_shape_x) + (y_c % _block_shape_y) * _block_shape_x) * batch_size;
                const int   in_x     = x_c / _block_shape_x;
                const int   in_y     = y_c / _block_shape_y;
                Coordinates input_coords{ 0, in_x, in_y, in_batch };
                memcpy(out.ptr(), _input->ptr_to_element(input_coords), element_size * _input->info()->dimension(0));
            },
            out);
            ++batch_id;
        }
        while(window.slide_window_slice_3D(slice_out));
    }
}
} // namespace arm_compute
