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
#include "arm_compute/core/CL/kernels/CLHOGDescriptorKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <set>
#include <sstream>
#include <string>

using namespace arm_compute;

CLHOGOrientationBinningKernel::CLHOGOrientationBinningKernel()
    : _input_magnitude(nullptr), _input_phase(nullptr), _output(nullptr), _cell_size()
{
}

void CLHOGOrientationBinningKernel::configure(const ICLTensor *input_magnitude, const ICLTensor *input_phase, ICLTensor *output, const HOGInfo *hog_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input_magnitude, 1, DataType::S16);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input_phase, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(hog_info == nullptr);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, hog_info->num_bins(), DataType::F32);
    ARM_COMPUTE_ERROR_ON(input_magnitude->info()->dimension(Window::DimX) != input_phase->info()->dimension(Window::DimX));
    ARM_COMPUTE_ERROR_ON(input_magnitude->info()->dimension(Window::DimY) != input_phase->info()->dimension(Window::DimY));

    _input_magnitude = input_magnitude;
    _input_phase     = input_phase;
    _output          = output;
    _cell_size       = hog_info->cell_size();

    float phase_scale = (PhaseType::SIGNED == hog_info->phase_type() ? hog_info->num_bins() / 360.0f : hog_info->num_bins() / 180.0f);
    phase_scale *= (PhaseType::SIGNED == hog_info->phase_type() ? 360.0f / 255.0f : 1.0f);

    std::stringstream args_str;
    args_str << "-DCELL_WIDTH=" << hog_info->cell_size().width << " ";
    args_str << "-DCELL_HEIGHT=" << hog_info->cell_size().height << " ";
    args_str << "-DNUM_BINS=" << hog_info->num_bins() << " ";
    args_str << "-DPHASE_SCALE=" << phase_scale << " ";

    // Construct kernel name
    std::set<std::string> build_opts = {};
    build_opts.insert(args_str.str());

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("hog_orientation_binning", build_opts));

    constexpr unsigned int num_elems_processed_per_iteration = 1;
    constexpr unsigned int num_elems_read_per_iteration      = 1;
    const unsigned int     num_rows_read_per_iteration       = hog_info->cell_size().height;
    constexpr unsigned int num_elems_written_per_iteration   = 1;

    // Configure kernel window
    Window                 win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win,
                              AccessWindowRectangle(input_magnitude->info(), 0, 0, num_elems_read_per_iteration, num_rows_read_per_iteration),
                              AccessWindowRectangle(input_phase->info(), 0, 0, num_elems_read_per_iteration, num_rows_read_per_iteration),
                              output_access);

    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    ICLKernel::configure(win);
}

void CLHOGOrientationBinningKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_2D();
    do
    {
        // Compute slice for the magnitude and phase tensors
        Window slice_mag_phase = window.first_slice_window_2D();
        slice_mag_phase.set(Window::DimX, Window::Dimension(window.x().start() * _cell_size.width, window.x().start() * _cell_size.width, _cell_size.width));
        slice_mag_phase.set(Window::DimY, Window::Dimension(window.y().start() * _cell_size.height, window.y().start() * _cell_size.height, _cell_size.height));

        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input_magnitude, slice_mag_phase);
        add_2D_tensor_argument(idx, _input_phase, slice_mag_phase);
        add_2D_tensor_argument(idx, _output, slice);

        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_2D(slice));
}

CLHOGBlockNormalizationKernel::CLHOGBlockNormalizationKernel()
    : _input(nullptr), _output(nullptr), _num_cells_per_block_stride()
{
}

void CLHOGBlockNormalizationKernel::configure(const ICLTensor *input, ICLTensor *output, const HOGInfo *hog_info)
{
    ARM_COMPUTE_ERROR_ON(hog_info == nullptr);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, hog_info->num_bins(), DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_NOT_IN(output, DataType::F32);

    // Number of cells per block
    const Size2D num_cells_per_block(hog_info->block_size().width / hog_info->cell_size().width,
                                     hog_info->block_size().height / hog_info->cell_size().height);

    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, hog_info->num_bins() * num_cells_per_block.area(), DataType::F32);

    // Number of cells per block stride
    const Size2D num_cells_per_block_stride(hog_info->block_stride().width / hog_info->cell_size().width,
                                            hog_info->block_stride().height / hog_info->cell_size().height);

    _input                      = input;
    _output                     = output;
    _num_cells_per_block_stride = num_cells_per_block_stride;

    std::stringstream args_str;
    args_str << "-DL2_HYST_THRESHOLD=" << hog_info->l2_hyst_threshold() << " ";
    args_str << "-DNUM_CELLS_PER_BLOCK_HEIGHT=" << num_cells_per_block.height << " ";
    args_str << "-DNUM_BINS_PER_BLOCK_X=" << num_cells_per_block.width *hog_info->num_bins() << " ";
    args_str << "-DNUM_BINS_PER_BLOCK=" << _output->info()->num_channels() << " ";
    args_str << "-DL2_NORM=" << static_cast<int>(HOGNormType::L2_NORM) << " ";
    args_str << "-DL1_NORM=" << static_cast<int>(HOGNormType::L1_NORM) << " ";
    args_str << "-DL2HYS_NORM=" << static_cast<int>(HOGNormType::L2HYS_NORM) << " ";
    args_str << "-DHOG_NORM_TYPE=" << static_cast<int>(hog_info->normalization_type()) << " ";

    // Construct kernel name
    std::set<std::string> build_opts = {};
    build_opts.insert(args_str.str());

    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("hog_block_normalization", build_opts));

    constexpr unsigned int num_elems_processed_per_iteration = 1;
    constexpr unsigned int num_elems_read_per_iteration      = 1;
    const unsigned int     num_rows_read_per_iteration       = num_cells_per_block.height;
    constexpr unsigned int num_elems_written_per_iteration   = 1;
    const unsigned int     num_rows_written_per_iteration    = num_cells_per_block.height;

    // Configure kernel window
    Window                win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowRectangle output_access(output->info(), 0, 0, num_elems_written_per_iteration, num_rows_written_per_iteration);

    update_window_and_padding(win,
                              AccessWindowRectangle(input->info(), 0, 0, num_elems_read_per_iteration, num_rows_read_per_iteration),
                              output_access);

    output_access.set_valid_region(win, input->info()->valid_region());

    ICLKernel::configure(win);
}

void CLHOGBlockNormalizationKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_2D();
    do
    {
        // Compute slice for the magnitude and phase tensors
        Window slice_in = window.first_slice_window_2D();
        slice_in.set_dimension_step(Window::DimX, _num_cells_per_block_stride.width);
        slice_in.set_dimension_step(Window::DimY, _num_cells_per_block_stride.height);

        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, slice_in);
        add_2D_tensor_argument(idx, _output, slice);

        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_2D(slice));
}
