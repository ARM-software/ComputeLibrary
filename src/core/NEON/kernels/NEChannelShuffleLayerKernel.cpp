/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "src/core/NEON/kernels/NEChannelShuffleLayerKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, unsigned int num_groups)
{
    // Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use Neon FP16 instructions.
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(input, DataLayout::NCHW, DataLayout::NHWC);

    const unsigned int channels = input->dimension(get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL));

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(num_groups < 2, "Channel shuffling with less than 2 groups would be inefficient");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(num_groups == channels, "Channel shuffling with same number of groups as number of channels would be inefficient");
    ARM_COMPUTE_RETURN_ERROR_ON(num_groups > channels); // There cannot be more groups than channels
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((channels % num_groups) != 0, "The number of channels must be a multiple of the number of groups");

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
    }

    return Status{};
}
void channel_shuffle_nhwc(const ITensor *input, ITensor *output, unsigned int num_groups, const Window &window)
{
    const DataLayout   data_layout = input->info()->data_layout();
    const unsigned int channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    const size_t       element_size = input->info()->element_size();
    const unsigned int K            = input->info()->dimension(channel_idx) / num_groups;
    const float        rK           = 1.f / K;

    Iterator in(input, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        // Shuffle channel
        const unsigned int curr_channel = id.x();
        const unsigned int group_id     = curr_channel * rK;
        const unsigned int r            = group_id * K;
        const unsigned int channel_id   = curr_channel - r;

        // Calculate output coordinates
        Coordinates out_coords = id;
        out_coords.set(Window::DimX, channel_id * num_groups + group_id);
        std::copy_n(in.ptr(), element_size, output->ptr_to_element(out_coords));
    },
    in);
}
void channel_shuffle_nchw(const ITensor *input, ITensor *output, unsigned int num_groups, const Window &window)
{
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    win.set(Window::DimY, Window::Dimension(0, 1, 1));

    const DataLayout   data_layout = input->info()->data_layout();
    const unsigned int width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    const unsigned int height          = input->info()->tensor_shape().y();
    const size_t       input_stride_y  = input->info()->strides_in_bytes().y();
    const size_t       output_stride_y = output->info()->strides_in_bytes().y();
    const size_t       row_size        = input->info()->dimension(width_idx) * input->info()->element_size();

    const unsigned int K  = input->info()->dimension(channel_idx) / num_groups;
    const float        rK = 1.f / K;

    Iterator in(input, win);

    execute_window_loop(win, [&](const Coordinates & id)
    {
        // Shuffle channel
        const unsigned int curr_channel = id.z();
        const unsigned int group_id     = curr_channel * rK;
        const unsigned int r            = group_id * K;
        const unsigned int channel_id   = curr_channel - r;

        // Calculate output coordinates
        Coordinates out_coords = id;
        out_coords.set(Window::DimZ, channel_id * num_groups + group_id);
        const uint8_t *input_ptr  = in.ptr();
        uint8_t       *output_ptr = output->ptr_to_element(out_coords);

        // Copy plane
        for(unsigned int y = 0; y < height; ++y)
        {
            std::copy_n(input_ptr, row_size, output_ptr);
            input_ptr += input_stride_y;
            output_ptr += output_stride_y;
        }
    },
    in);
}
} // namespace

NEChannelShuffleLayerKernel::NEChannelShuffleLayerKernel()
    : _input(nullptr), _output(nullptr), _num_groups()
{
}

void NEChannelShuffleLayerKernel::configure(const ITensor *input, ITensor *output, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), *input->info()->clone());

    _input      = input;
    _output     = output;
    _num_groups = num_groups;

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), num_groups));

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps());

    // The NEChannelShuffleLayerKernel doesn't need padding so update_window_and_padding() can be skipped
    Coordinates coord;
    coord.set_num_dimensions(output->info()->num_dimensions());
    output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));

    INEKernel::configure(win);
}

Status NEChannelShuffleLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, unsigned int num_groups)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, num_groups));
    return Status{};
}

void NEChannelShuffleLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    switch(_input->info()->data_layout())
    {
        case DataLayout::NHWC:
            channel_shuffle_nhwc(_input, _output, _num_groups, window);
            break;
        case DataLayout::NCHW:
            channel_shuffle_nchw(_input, _output, _num_groups, window);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data layout!");
            break;
    }
}
} // namespace arm_compute
