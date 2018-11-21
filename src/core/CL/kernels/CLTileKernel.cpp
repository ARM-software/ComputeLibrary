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
#include "arm_compute/core/CL/kernels/CLTileKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const Multiples &multiples)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(multiples.size() > 4);
    ARM_COMPUTE_RETURN_ERROR_ON(multiples.empty());
    ARM_COMPUTE_RETURN_ERROR_ON(std::any_of(multiples.begin(), multiples.end(), [](uint32_t e)
    {
        return e == 0;
    }));

    // Validate output if initialized
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(misc::shape_calculator::compute_tiled_shape(input->tensor_shape(), multiples), output->tensor_shape());
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}
} // namespace

CLTileKernel::CLTileKernel()
    : _input(nullptr), _output(nullptr)
{
}

void CLTileKernel::configure(const ICLTensor *input, ICLTensor *output, const Multiples &multiples)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Auto initialize output
    TensorShape tiled_shape = misc::shape_calculator::compute_tiled_shape(input->info()->tensor_shape(), multiples);
    auto_init_if_empty(*output->info(), tiled_shape, 1, input->info()->data_type());

    // Validate
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), multiples));

    _input  = input;
    _output = output;

    const DataType     data_type      = input->info()->data_type();
    const int          vec_size_x     = 16 / input->info()->element_size();
    const int          input_width_x  = input->info()->tensor_shape().x();
    const unsigned int offset         = ceil_to_multiple(input_width_x, vec_size_x) - input_width_x;
    const bool         multi_access_x = (input_width_x / vec_size_x > 0);

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_opts.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(input_width_x));
    build_opts.add_option("-DSRC_HEIGHT=" + support::cpp11::to_string(input->info()->dimension(1)));
    build_opts.add_option("-DSRC_DEPTH=" + support::cpp11::to_string(input->info()->dimension(2)));
    build_opts.add_option("-DSRC_BATCHES=" + support::cpp11::to_string(input->info()->dimension(3)));
    build_opts.add_option("-DDST_DEPTH=" + support::cpp11::to_string(output->info()->dimension(2)));
    build_opts.add_option_if(multi_access_x, "-DOFFSET=" + support::cpp11::to_string(offset));
    build_opts.add_option_if(multi_access_x, "-DVEC_SIZE=" + support::cpp11::to_string(vec_size_x));
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("tile", build_opts.options()));

    // Configure window without padding
    Window win = calculate_max_window(*output->info());

    if(multi_access_x)
    {
        // If multi-access is enabled, no thread should cross the tile boundaries. This means we need
        // as many threads as those to cover a single tile times multiples[0]. Note that if threads
        // do not cross the boundaries of the tiles, they won't cross the boundaries of the last tile, and
        // we don't need to pad the output
        const unsigned int size_win_x = ceil_to_multiple(input->info()->dimension(0), vec_size_x) * multiples[0];
        win.set(Window::DimX,
                Window::Dimension(win.x().start(), size_win_x, vec_size_x));
    }

    ICLKernel::configure_internal(win);

    // Set config_id for enabling LWS tuning
    _config_id = "tile";
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    for(unsigned int i = 0; i < multiples.size(); ++i)
    {
        _config_id += "_";
        _config_id += support::cpp11::to_string(input->info()->dimension(i));
        _config_id += "_";
        _config_id += support::cpp11::to_string(multiples[i]);
    }
}

Status CLTileKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const Multiples &multiples)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, multiples));
    return Status{};
}

void CLTileKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice     = collapsed.first_slice_window_4D();

    do
    {
        unsigned int idx = 0;
        add_4D_tensor_argument(idx, _input, slice);
        add_4D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice);
    }
    while(collapsed.slide_window_slice_4D(slice));
}
} // namespace arm_compute
