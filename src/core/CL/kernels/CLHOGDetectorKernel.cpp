/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLHOGDetectorKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLHOG.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

using namespace arm_compute;

CLHOGDetectorKernel::CLHOGDetectorKernel()
    : _input(nullptr), _detection_windows(), _num_detection_windows(nullptr)
{
}

void CLHOGDetectorKernel::configure(const ICLTensor *input, const ICLHOG *hog, ICLDetectionWindowArray *detection_windows, cl::Buffer *num_detection_windows, const Size2D &detection_window_stride,
                                    float threshold, uint16_t idx_class)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_NOT_IN(input, DataType::F32);
    ARM_COMPUTE_ERROR_ON(hog == nullptr);
    ARM_COMPUTE_ERROR_ON(detection_windows == nullptr);
    ARM_COMPUTE_ERROR_ON(num_detection_windows == nullptr);
    ARM_COMPUTE_ERROR_ON((detection_window_stride.width % hog->info()->block_stride().width) != 0);
    ARM_COMPUTE_ERROR_ON((detection_window_stride.height % hog->info()->block_stride().height) != 0);

    const Size2D &detection_window_size = hog->info()->detection_window_size();
    const Size2D &block_size            = hog->info()->block_size();
    const Size2D &block_stride          = hog->info()->block_stride();

    _input                 = input;
    _detection_windows     = detection_windows;
    _num_detection_windows = num_detection_windows;

    const unsigned int num_bins_per_descriptor_x   = ((detection_window_size.width - block_size.width) / block_stride.width + 1) * input->info()->num_channels();
    const unsigned int num_blocks_per_descriptor_y = (detection_window_size.height - block_size.height) / block_stride.height + 1;

    ARM_COMPUTE_ERROR_ON((num_bins_per_descriptor_x * num_blocks_per_descriptor_y + 1) != hog->info()->descriptor_size());

    std::stringstream args_str;
    args_str << "-DNUM_BLOCKS_PER_DESCRIPTOR_Y=" << num_blocks_per_descriptor_y << " ";
    args_str << "-DNUM_BINS_PER_DESCRIPTOR_X=" << num_bins_per_descriptor_x << " ";
    args_str << "-DTHRESHOLD=" << threshold << " ";
    args_str << "-DMAX_NUM_DETECTION_WINDOWS=" << detection_windows->max_num_values() << " ";
    args_str << "-DIDX_CLASS=" << idx_class << " ";
    args_str << "-DDETECTION_WINDOW_WIDTH=" << detection_window_size.width << " ";
    args_str << "-DDETECTION_WINDOW_HEIGHT=" << detection_window_size.height << " ";
    args_str << "-DDETECTION_WINDOW_STRIDE_WIDTH=" << detection_window_stride.width << " ";
    args_str << "-DDETECTION_WINDOW_STRIDE_HEIGHT=" << detection_window_stride.height << " ";

    // Construct kernel name
    std::set<std::string> build_opts = {};
    build_opts.insert(args_str.str());

    // Create kernel
    const std::string kernel_name = std::string("hog_detector");
    _kernel                       = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts));

    // Set static kernel arguments
    unsigned int idx = num_arguments_per_2D_tensor(); // Skip the input parameters
    _kernel.setArg(idx++, hog->cl_buffer());
    _kernel.setArg(idx++, detection_windows->cl_buffer());
    _kernel.setArg(idx++, *_num_detection_windows);

    // Get the number of blocks along the x and y directions of the input tensor
    const ValidRegion &valid_region = input->info()->valid_region();
    const size_t       num_blocks_x = valid_region.shape[0];
    const size_t       num_blocks_y = valid_region.shape[1];

    // Get the number of blocks along the x and y directions of the detection window
    const size_t num_blocks_per_detection_window_x = detection_window_size.width / block_stride.width;
    const size_t num_blocks_per_detection_window_y = detection_window_size.height / block_stride.height;

    const size_t window_step_x = detection_window_stride.width / block_stride.width;
    const size_t window_step_y = detection_window_stride.height / block_stride.height;

    // Configure kernel window
    Window win;
    win.set(Window::DimX, Window::Dimension(0, floor_to_multiple(num_blocks_x - num_blocks_per_detection_window_x, window_step_x) + window_step_x, window_step_x));
    win.set(Window::DimY, Window::Dimension(0, floor_to_multiple(num_blocks_y - num_blocks_per_detection_window_y, window_step_y) + window_step_y, window_step_y));

    constexpr unsigned int num_elems_read_per_iteration = 1;
    const unsigned int     num_rows_read_per_iteration  = num_blocks_per_descriptor_y;

    update_window_and_padding(win, AccessWindowRectangle(input->info(), 0, 0, num_elems_read_per_iteration, num_rows_read_per_iteration));

    ICLKernel::configure_internal(win);

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
}

void CLHOGDetectorKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_2D();
    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, slice);

        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_2D(slice));
}
