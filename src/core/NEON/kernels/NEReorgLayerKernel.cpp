/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEReorgLayerKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <cstddef>
#include <cstdint>

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, int32_t stride)
{
    //Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use NEON FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1,
                                                         DataType::U8, DataType::S8, DataType::QASYMM8,
                                                         DataType::U16, DataType::S16,
                                                         DataType::U32, DataType::S32,
                                                         DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_layout() == DataLayout::UNKNOWN);

    const size_t idx_width  = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_height = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);

    ARM_COMPUTE_RETURN_ERROR_ON(stride <= 0);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((input->tensor_shape()[idx_width] % stride) != 0, "The width of the input tensor must be a multiple of stride");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((input->tensor_shape()[idx_height] % stride) != 0, "The height of the input tensor must be a multiple of stride");

    // Validate output if initialized
    if(output->total_size() != 0)
    {
        const TensorInfo tensor_info_output = output->clone()->set_tensor_shape(misc::shape_calculator::compute_reorg_output_shape(*input, stride));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &tensor_info_output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}
} // namespace

NEReorgLayerKernel::NEReorgLayerKernel()
    : _input(nullptr), _output(nullptr), _stride(1)
{
}

void NEReorgLayerKernel::configure(const ITensor *input, ITensor *output, int32_t stride)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Output auto inizialitation if not yet initialized
    const TensorShape output_shape = misc::shape_calculator::compute_reorg_output_shape(*input->info(), stride);
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape));

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), stride));

    _input  = input;
    _output = output;
    _stride = stride;

    // The NEReorgLayerKernel doesn't need padding so update_window_and_padding() can be skipped
    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps());

    ICPPKernel::configure(win);
}

Status NEReorgLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, int32_t stride)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, stride));
    return Status{};
}

void NEReorgLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICPPKernel::window(), window);

    const DataLayout data_layout = _input->info()->data_layout();
    const size_t     idx_w       = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const size_t     idx_h       = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const size_t     idx_c       = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    const unsigned int stride = _stride;
    const unsigned int out_c  = _output->info()->tensor_shape()[idx_c] / (stride * stride);
    const uint8_t     *in_ptr = _input->buffer();

    // Collapse
    Window collapsed_window = window.collapse_if_possible(window, 4);

    // Create Iterator
    Iterator out(_output, collapsed_window);

    // Perform reorg
    execute_window_loop(collapsed_window, [&](const Coordinates & id)
    {
        // Get spatial coords and channels
        const unsigned int w = id[idx_w];
        const unsigned int h = id[idx_h];
        const unsigned int c = id[idx_c];

        // Calculate mapping
        const unsigned int offset     = c / out_c;
        Coordinates        map_coords = id;
        map_coords.set(idx_w, w * stride + offset % stride);
        map_coords.set(idx_h, h * stride + offset / stride);
        map_coords.set(idx_c, c % out_c);

        // Perform mapping
        std::memcpy(out.ptr(), in_ptr + _input->info()->offset_element_in_bytes(map_coords), _input->info()->element_size());
    },
    out);
}
} // namespace arm_compute
