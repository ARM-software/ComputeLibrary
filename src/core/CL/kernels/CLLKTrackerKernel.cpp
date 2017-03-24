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
#include "arm_compute/core/CL/kernels/CLLKTrackerKernel.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLArray.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <cmath>

using namespace arm_compute;

void CLLKTrackerInitKernel::configure(const ICLKeyPointArray *old_points, const ICLKeyPointArray *new_points_estimates,
                                      ICLLKInternalKeypointArray *old_points_internal, ICLLKInternalKeypointArray *new_points_internal,
                                      bool use_initial_estimate, size_t level, size_t num_levels, float pyramid_scale)

{
    ARM_COMPUTE_ERROR_ON(old_points == nullptr);
    ARM_COMPUTE_ERROR_ON(old_points_internal == nullptr);
    ARM_COMPUTE_ERROR_ON(new_points_internal == nullptr);

    const float scale = std::pow(pyramid_scale, level);

    // Create kernel
    std::string kernel_name = "init_level";
    if(level == (num_levels - 1))
    {
        kernel_name += (use_initial_estimate) ? std::string("_max_initial_estimate") : std::string("_max");
    }
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name));

    // Set arguments
    unsigned int idx = 0;
    if(level == (num_levels - 1))
    {
        _kernel.setArg(idx++, old_points->cl_buffer());
        if(use_initial_estimate)
        {
            _kernel.setArg(idx++, new_points_estimates->cl_buffer());
        }
    }
    _kernel.setArg(idx++, old_points_internal->cl_buffer());
    _kernel.setArg(idx++, new_points_internal->cl_buffer());
    _kernel.setArg<float>(idx++, scale);

    // Set window
    Window window;
    window.set(Window::DimX, Window::Dimension(0, old_points->num_values(), 1));
    window.set(Window::DimY, Window::Dimension(0, 1, 1));
    ICLKernel::configure(window);
}

void CLLKTrackerInitKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    enqueue(queue, *this, window);
}

void CLLKTrackerFinalizeKernel::configure(ICLLKInternalKeypointArray *new_points_internal, ICLKeyPointArray *new_points)

{
    ARM_COMPUTE_ERROR_ON(new_points_internal == nullptr);
    ARM_COMPUTE_ERROR_ON(new_points == nullptr);

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("finalize"));

    // Set arguments
    unsigned int idx = 0;
    _kernel.setArg(idx++, new_points_internal->cl_buffer());
    _kernel.setArg(idx++, new_points->cl_buffer());

    // Set window
    Window window;
    window.set(Window::DimX, Window::Dimension(0, new_points_internal->num_values(), 1));
    window.set(Window::DimY, Window::Dimension(0, 1, 1));
    ICLKernel::configure(window);
}

void CLLKTrackerFinalizeKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    enqueue(queue, *this, window);
}

void CLLKTrackerStage0Kernel::configure(const ICLTensor *old_input, const ICLTensor *old_scharr_gx, const ICLTensor *old_scharr_gy,
                                        ICLLKInternalKeypointArray *old_points_internal, ICLLKInternalKeypointArray *new_points_internal,
                                        ICLCoefficientTableArray *coeff_table, ICLOldValArray *old_ival,
                                        size_t window_dimension, size_t level, int32_t border_offset)

{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(old_input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(old_scharr_gx, 1, DataType::S16);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(old_scharr_gy, 1, DataType::S16);
    ARM_COMPUTE_ERROR_ON(old_points_internal == nullptr);
    ARM_COMPUTE_ERROR_ON(new_points_internal == nullptr);
    ARM_COMPUTE_ERROR_ON(coeff_table == nullptr);
    ARM_COMPUTE_ERROR_ON(old_ival == nullptr);

    // Set window
    Window window;
    window.set(Window::DimX, Window::Dimension(0, new_points_internal->num_values(), 1));
    window.set(Window::DimY, Window::Dimension(0, 1, 1));
    ICLKernel::configure(window);

    // Initialize required variables
    const int       level0           = (level == 0) ? 1 : 0;
    const int       window_size      = window_dimension;
    const int       window_size_pow2 = window_dimension * window_dimension;
    const int       window_size_half = window_dimension / 2;
    float           eig_const        = 1.0f / (2.0f * window_dimension * window_dimension);
    const cl_float3 border_limits =
    {
        {
            static_cast<float>(old_input->info()->dimension(0)) - window_size - border_offset - 1,
            static_cast<float>(old_input->info()->dimension(1)) - window_size - border_offset - 1,
            static_cast<float>(border_offset)
        }
    };

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("lktracker_stage0"));

    // Set arguments
    unsigned int idx = 0;
    add_2D_tensor_argument(idx, old_input, window);
    add_2D_tensor_argument(idx, old_scharr_gx, window);
    add_2D_tensor_argument(idx, old_scharr_gy, window);
    _kernel.setArg(idx++, old_points_internal->cl_buffer());
    _kernel.setArg(idx++, new_points_internal->cl_buffer());
    _kernel.setArg(idx++, coeff_table->cl_buffer());
    _kernel.setArg(idx++, old_ival->cl_buffer());
    _kernel.setArg<int>(idx++, window_size);
    _kernel.setArg<int>(idx++, window_size_pow2);
    _kernel.setArg<int>(idx++, window_size_half);
    _kernel.setArg<cl_float3>(idx++, border_limits);
    _kernel.setArg<float>(idx++, eig_const);
    _kernel.setArg<int>(idx++, level0);
}

void CLLKTrackerStage0Kernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    enqueue(queue, *this, window);
}

void CLLKTrackerStage1Kernel::configure(const ICLTensor *new_input, ICLLKInternalKeypointArray *new_points_internal, ICLCoefficientTableArray *coeff_table, ICLOldValArray *old_ival,
                                        Termination termination, float epsilon, size_t num_iterations, size_t window_dimension, size_t level, int32_t border_offset)

{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(new_input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(new_points_internal == nullptr);
    ARM_COMPUTE_ERROR_ON(coeff_table == nullptr);
    ARM_COMPUTE_ERROR_ON(old_ival == nullptr);

    // Set window
    Window window;
    window.set(Window::DimX, Window::Dimension(0, new_points_internal->num_values(), 1));
    window.set(Window::DimY, Window::Dimension(0, 1, 1));
    ICLKernel::configure(window);

    // Initialize required variables
    const int       level0           = (level == 0) ? 1 : 0;
    const int       window_size      = window_dimension;
    const int       window_size_pow2 = window_dimension * window_dimension;
    const int       window_size_half = window_dimension / 2;
    float           eig_const        = 1.0f / (2.0f * window_dimension * window_dimension);
    const cl_float3 border_limits =
    {
        {
            static_cast<float>(new_input->info()->dimension(0)) - window_size - border_offset - 1,
            static_cast<float>(new_input->info()->dimension(1)) - window_size - border_offset - 1,
            static_cast<float>(border_offset)
        }
    };
    const int term_iteration = (termination == Termination::TERM_CRITERIA_ITERATIONS || termination == Termination::TERM_CRITERIA_BOTH) ? 1 : 0;
    const int term_epsilon   = (termination == Termination::TERM_CRITERIA_EPSILON || termination == Termination::TERM_CRITERIA_BOTH) ? 1 : 0;

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("lktracker_stage1"));

    // Set arguments
    unsigned int idx = 0;
    add_2D_tensor_argument(idx, new_input, window);
    _kernel.setArg(idx++, new_points_internal->cl_buffer());
    _kernel.setArg(idx++, coeff_table->cl_buffer());
    _kernel.setArg(idx++, old_ival->cl_buffer());
    _kernel.setArg<int>(idx++, window_size);
    _kernel.setArg<int>(idx++, window_size_pow2);
    _kernel.setArg<int>(idx++, window_size_half);
    _kernel.setArg<int>(idx++, num_iterations);
    _kernel.setArg<float>(idx++, epsilon);
    _kernel.setArg<cl_float3>(idx++, border_limits);
    _kernel.setArg<float>(idx++, eig_const);
    _kernel.setArg<int>(idx++, level0);
    _kernel.setArg<int>(idx++, term_iteration);
    _kernel.setArg<int>(idx++, term_epsilon);
}

void CLLKTrackerStage1Kernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    enqueue(queue, *this, window);
}
