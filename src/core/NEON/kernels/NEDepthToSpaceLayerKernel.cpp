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
#include "src/core/NEON/kernels/NEDepthToSpaceLayerKernel.h"

#include "arm_compute/core/CoreTypes.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/Validate.h"

#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/depth_to_space/list.h"

#include <cstdint>

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, int32_t block_shape)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4);
    ARM_COMPUTE_RETURN_ERROR_ON(block_shape < 2);

    const DataLayout data_layout = input->data_layout();
    const int        idx_channel = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);
    ARM_COMPUTE_RETURN_ERROR_ON(input->tensor_shape()[idx_channel] % (block_shape * block_shape) != 0);
    // Validate output if initialized
    if (output->total_size() != 0)
    {
        const int idx_width  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
        const int idx_height = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
        ARM_COMPUTE_RETURN_ERROR_ON(output->tensor_shape()[idx_width] !=
                                    (block_shape * input->tensor_shape()[idx_width]));
        ARM_COMPUTE_RETURN_ERROR_ON(output->tensor_shape()[idx_height] !=
                                    (block_shape * input->tensor_shape()[idx_height]));
        ARM_COMPUTE_RETURN_ERROR_ON(output->num_dimensions() > 4);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}
} // namespace

NEDepthToSpaceLayerKernel::NEDepthToSpaceLayerKernel()
    : _input(nullptr),
      _output(nullptr),
      _block_shape(),
      _data_layout(DataLayout::UNKNOWN),
      _split_dimension(Window::DimY)
{
}

void NEDepthToSpaceLayerKernel::configure(const ITensor *input, ITensor *output, int32_t block_shape)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    TensorShape output_shape = misc::shape_calculator::compute_depth_to_space_shape(
        input->info()->tensor_shape(), input->info()->data_layout(), block_shape);
    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape));

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), block_shape));

    _input       = input;
    _output      = output;
    _block_shape = block_shape;
    _data_layout = input->info()->data_layout();

    constexpr size_t dim_b = 3;
    const auto       dim_h = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);
    const auto       dim_w = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
    const auto       dim_c = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::CHANNEL);

    ARM_COMPUTE_ERROR_ON(get_data_layout_dimension_index(_data_layout, DataLayoutDimension::BATCHES) != dim_b);

    // Configure kernel window
    Steps steps;
    steps.set(dim_h, block_shape);
    steps.set(dim_w, block_shape);
    steps.set(dim_c, output->info()->dimension(dim_c));

    Window win = calculate_max_window(*output->info(), steps);
    ICPPKernel::configure(win);

    const auto num_batches = input->info()->tensor_shape().total_size_upper(dim_b);
    if (num_batches > 1)
    {
        _split_dimension = dim_b;
    }
    else
    {
        _split_dimension = dim_h;
    }
}

Status NEDepthToSpaceLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, int32_t block_shape)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, block_shape));
    return Status{};
}

size_t NEDepthToSpaceLayerKernel::get_split_dimension() const
{
    return _split_dimension;
}

void NEDepthToSpaceLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICPPKernel::window(), window);

    const auto *input_info  = _input->info();
    const auto *output_info = _output->info();

    const auto  element_size   = input_info->element_size();
    const auto &input_strides  = input_info->strides_in_bytes();
    const auto &output_strides = output_info->strides_in_bytes();

    const auto &input_shape = input_info->tensor_shape();

    const uintptr_t k_input_strides[]  = {input_strides[0], input_strides[1], input_strides[2], input_strides[3]};
    const uintptr_t k_output_strides[] = {output_strides[0], output_strides[1], output_strides[2], output_strides[3]};

    const uint8_t *k_input_ptr  = _input->buffer();
    uint8_t       *k_output_ptr =               //
        _output->buffer() +                     //
        window[3].start() * output_strides[3] + //
        window[2].start() * output_strides[2] + //
        window[1].start() * output_strides[1] + //
        window[0].start() * output_strides[0];

    if (_data_layout == DataLayout::NCHW)
    {
        ARM_COMPUTE_ERROR_ON_MSG(window[2].start() != 0 || window[2].end() != window[2].step(),
                                 "The window cannot be splitted in channel dimension");

        const uintptr_t k_input_shape[] = {
            window.num_iterations(0), //
            window.num_iterations(1), //
            input_shape[2],           // The window cannot be splitted in channel dimension.
            window.num_iterations(3)  //
        };

        k_input_ptr += window[3].start() * input_strides[3] +                               //
                       window[2].start() * _block_shape * _block_shape * input_strides[2] + //
                       (window[1].start() / _block_shape) * input_strides[1] +              //
                       (window[0].start() / _block_shape) * input_strides[0];

        cpu::depth_to_space_nchw_any(                         //
            k_input_ptr, k_output_ptr,                        //
            k_input_shape, k_input_strides, k_output_strides, //
            element_size, _block_shape);
    }
    else
    {
        ARM_COMPUTE_ERROR_ON_MSG(window[0].start() != 0 || window[0].end() != window[0].step(),
                                 "The window cannot be splitted in channel dimension");

        const uintptr_t k_input_shape[] = {
            input_shape[0],           // The window cannot be splitted in channel dimension.
            window.num_iterations(1), //
            window.num_iterations(2), //
            window.num_iterations(3)  //
        };

        k_input_ptr += window[3].start() * input_strides[3] +                  //
                       (window[2].start() / _block_shape) * input_strides[2] + //
                       (window[1].start() / _block_shape) * input_strides[1] + //
                       window[0].start() * _block_shape * _block_shape * input_strides[0];

        cpu::depth_to_space_nhwc_any(                         //
            k_input_ptr, k_output_ptr,                        //
            k_input_shape, k_input_strides, k_output_strides, //
            element_size, _block_shape);
    }
}
} // namespace arm_compute
