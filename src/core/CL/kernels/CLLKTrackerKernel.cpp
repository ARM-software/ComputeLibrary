/*
 * Copyright (c) 2017-2020 ARM Limited.
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

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLArray.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <cmath>

using namespace arm_compute;

void CLLKTrackerInitKernel::configure(const ICLKeyPointArray *old_points, const ICLKeyPointArray *new_points_estimates,
                                      ICLLKInternalKeypointArray *old_points_internal, ICLLKInternalKeypointArray *new_points_internal,
                                      bool use_initial_estimate, size_t level, size_t num_levels, float pyramid_scale)
{
    configure(CLKernelLibrary::get().get_compile_context(), old_points, new_points_estimates, old_points_internal, new_points_internal, use_initial_estimate, level, num_levels, pyramid_scale);
}

void CLLKTrackerInitKernel::configure(CLCompileContext &compile_context, const ICLKeyPointArray *old_points, const ICLKeyPointArray *new_points_estimates,
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
    _kernel = create_kernel(compile_context, kernel_name);

    // Set static kernel arguments
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
    _kernel.setArg<cl_float>(idx++, scale);

    // Configure kernel window
    Window window;
    window.set(Window::DimX, Window::Dimension(0, old_points->num_values(), 1));
    window.set(Window::DimY, Window::Dimension(0, 1, 1));
    ICLKernel::configure_internal(window);
}

void CLLKTrackerInitKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    enqueue(queue, *this, window, lws_hint());
}

void CLLKTrackerFinalizeKernel::configure(ICLLKInternalKeypointArray *new_points_internal, ICLKeyPointArray *new_points)
{
    configure(CLKernelLibrary::get().get_compile_context(), new_points_internal, new_points);
}

void CLLKTrackerFinalizeKernel::configure(CLCompileContext &compile_context, ICLLKInternalKeypointArray *new_points_internal, ICLKeyPointArray *new_points)

{
    ARM_COMPUTE_ERROR_ON(new_points_internal == nullptr);
    ARM_COMPUTE_ERROR_ON(new_points == nullptr);

    // Create kernel
    _kernel = create_kernel(compile_context, "finalize");

    // Set static kernel arguments
    unsigned int idx = 0;
    _kernel.setArg(idx++, new_points_internal->cl_buffer());
    _kernel.setArg(idx++, new_points->cl_buffer());

    // Configure kernel window
    Window window;
    window.set(Window::DimX, Window::Dimension(0, new_points_internal->num_values(), 1));
    window.set(Window::DimY, Window::Dimension(0, 1, 1));
    ICLKernel::configure_internal(window);
}

void CLLKTrackerFinalizeKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    enqueue(queue, *this, window, lws_hint());
}

CLLKTrackerStage0Kernel::CLLKTrackerStage0Kernel()
    : _old_input(nullptr), _old_scharr_gx(nullptr), _old_scharr_gy(nullptr)
{
}

void CLLKTrackerStage0Kernel::configure(const ICLTensor *old_input, const ICLTensor *old_scharr_gx, const ICLTensor *old_scharr_gy,
                                        ICLLKInternalKeypointArray *old_points_internal, ICLLKInternalKeypointArray *new_points_internal,
                                        ICLCoefficientTableArray *coeff_table, ICLOldValArray *old_ival,
                                        size_t window_dimension, size_t level)
{
    configure(CLKernelLibrary::get().get_compile_context(), old_input, old_scharr_gx, old_scharr_gy, old_points_internal, new_points_internal, coeff_table, old_ival, window_dimension, level);
}

void CLLKTrackerStage0Kernel::configure(CLCompileContext &compile_context, const ICLTensor *old_input, const ICLTensor *old_scharr_gx, const ICLTensor *old_scharr_gy,
                                        ICLLKInternalKeypointArray *old_points_internal, ICLLKInternalKeypointArray *new_points_internal,
                                        ICLCoefficientTableArray *coeff_table, ICLOldValArray *old_ival,
                                        size_t window_dimension, size_t level)

{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(old_input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(old_scharr_gx, 1, DataType::S16);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(old_scharr_gy, 1, DataType::S16);
    ARM_COMPUTE_ERROR_ON(old_points_internal == nullptr);
    ARM_COMPUTE_ERROR_ON(new_points_internal == nullptr);
    ARM_COMPUTE_ERROR_ON(coeff_table == nullptr);
    ARM_COMPUTE_ERROR_ON(old_ival == nullptr);

    _old_input     = old_input;
    _old_scharr_gx = old_scharr_gx;
    _old_scharr_gy = old_scharr_gy;

    // Configure kernel window
    Window window;
    window.set(Window::DimX, Window::Dimension(0, new_points_internal->num_values(), 1));
    window.set(Window::DimY, Window::Dimension(0, 1, 1));

    const ValidRegion valid_region = intersect_valid_regions(
                                         old_input->info()->valid_region(),
                                         old_scharr_gx->info()->valid_region(),
                                         old_scharr_gy->info()->valid_region());

    update_window_and_padding(window,
                              AccessWindowStatic(old_input->info(), valid_region.start(0), valid_region.start(1),
                                                 valid_region.end(0), valid_region.end(1)),
                              AccessWindowStatic(old_scharr_gx->info(), valid_region.start(0), valid_region.start(1),
                                                 valid_region.end(0), valid_region.end(1)),
                              AccessWindowStatic(old_scharr_gy->info(), valid_region.start(0), valid_region.start(1),
                                                 valid_region.end(0), valid_region.end(1)));

    ICLKernel::configure_internal(window);

    // Initialize required variables
    const int       level0              = (level == 0) ? 1 : 0;
    const int       window_size         = window_dimension;
    const int       window_size_squared = window_dimension * window_dimension;
    const int       window_size_half    = window_dimension / 2;
    const float     eig_const           = 1.0f / (2.0f * window_size_squared);
    const cl_float3 border_limits =
    {
        {
            // -1 because we load 2 values at once for bilinear interpolation
            static_cast<cl_float>(valid_region.end(0) - window_size - 1),
            static_cast<cl_float>(valid_region.end(1) - window_size - 1),
            static_cast<cl_float>(valid_region.start(0))
        }
    };

    // Create kernel
    _kernel = create_kernel(compile_context, "lktracker_stage0");

    // Set arguments
    unsigned int idx = 3 * num_arguments_per_2D_tensor();
    _kernel.setArg(idx++, old_points_internal->cl_buffer());
    _kernel.setArg(idx++, new_points_internal->cl_buffer());
    _kernel.setArg(idx++, coeff_table->cl_buffer());
    _kernel.setArg(idx++, old_ival->cl_buffer());
    _kernel.setArg<cl_int>(idx++, window_size);
    _kernel.setArg<cl_int>(idx++, window_size_squared);
    _kernel.setArg<cl_int>(idx++, window_size_half);
    _kernel.setArg<cl_float3>(idx++, border_limits);
    _kernel.setArg<cl_float>(idx++, eig_const);
    _kernel.setArg<cl_int>(idx++, level0);
}

void CLLKTrackerStage0Kernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    // Set static tensor arguments. Setting here as allocation might be deferred.
    unsigned int idx = 0;
    add_2D_tensor_argument(idx, _old_input, window);
    add_2D_tensor_argument(idx, _old_scharr_gx, window);
    add_2D_tensor_argument(idx, _old_scharr_gy, window);

    enqueue(queue, *this, window, lws_hint());
}

CLLKTrackerStage1Kernel::CLLKTrackerStage1Kernel()
    : _new_input(nullptr)
{
}

void CLLKTrackerStage1Kernel::configure(const ICLTensor *new_input, ICLLKInternalKeypointArray *new_points_internal, ICLCoefficientTableArray *coeff_table, ICLOldValArray *old_ival,
                                        Termination termination, float epsilon, size_t num_iterations, size_t window_dimension, size_t level)
{
    configure(CLKernelLibrary::get().get_compile_context(), new_input, new_points_internal, coeff_table, old_ival, termination, epsilon, num_iterations, window_dimension, level);
}

void CLLKTrackerStage1Kernel::configure(CLCompileContext &compile_context, const ICLTensor *new_input, ICLLKInternalKeypointArray *new_points_internal, ICLCoefficientTableArray *coeff_table,
                                        ICLOldValArray *old_ival,
                                        Termination termination, float epsilon, size_t num_iterations, size_t window_dimension, size_t level)

{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(new_input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(new_points_internal == nullptr);
    ARM_COMPUTE_ERROR_ON(coeff_table == nullptr);
    ARM_COMPUTE_ERROR_ON(old_ival == nullptr);

    _new_input = new_input;

    // Configure kernel window
    Window window;
    window.set(Window::DimX, Window::Dimension(0, new_points_internal->num_values(), 1));
    window.set(Window::DimY, Window::Dimension(0, 1, 1));

    const ValidRegion &valid_region = new_input->info()->valid_region();

    update_window_and_padding(window,
                              AccessWindowStatic(new_input->info(), valid_region.start(0), valid_region.start(1),
                                                 valid_region.end(0), valid_region.end(1)));

    ICLKernel::configure_internal(window);

    // Initialize required variables
    const int       level0              = (level == 0) ? 1 : 0;
    const int       window_size         = window_dimension;
    const int       window_size_squared = window_dimension * window_dimension;
    const int       window_size_half    = window_dimension / 2;
    const float     eig_const           = 1.0f / (2.0f * window_size_squared);
    const cl_float3 border_limits =
    {
        {
            // -1 because we load 2 values at once for bilinear interpolation
            static_cast<cl_float>(valid_region.end(0) - window_size - 1),
            static_cast<cl_float>(valid_region.end(1) - window_size - 1),
            static_cast<cl_float>(valid_region.start(0))
        }
    };

    // Set maximum number of iterations used for convergence
    const size_t max_iterations = 1000;
    num_iterations              = (termination == Termination::TERM_CRITERIA_EPSILON) ? max_iterations : num_iterations;

    const int term_epsilon = (termination == Termination::TERM_CRITERIA_EPSILON || termination == Termination::TERM_CRITERIA_BOTH) ? 1 : 0;

    // Create kernel
    _kernel = create_kernel(compile_context, "lktracker_stage1");

    // Set static kernel arguments
    unsigned int idx = num_arguments_per_2D_tensor();
    _kernel.setArg(idx++, new_points_internal->cl_buffer());
    _kernel.setArg(idx++, coeff_table->cl_buffer());
    _kernel.setArg(idx++, old_ival->cl_buffer());
    _kernel.setArg<cl_int>(idx++, window_size);
    _kernel.setArg<cl_int>(idx++, window_size_squared);
    _kernel.setArg<cl_int>(idx++, window_size_half);
    _kernel.setArg<cl_int>(idx++, num_iterations);
    _kernel.setArg<cl_float>(idx++, epsilon);
    _kernel.setArg<cl_float3>(idx++, border_limits);
    _kernel.setArg<cl_float>(idx++, eig_const);
    _kernel.setArg<cl_int>(idx++, level0);
    _kernel.setArg<cl_int>(idx++, term_epsilon);
}

void CLLKTrackerStage1Kernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    // Set static tensor arguments. Setting here as allocation might be deferred.
    unsigned int idx = 0;
    add_2D_tensor_argument(idx, _new_input, window);

    enqueue(queue, *this, window, lws_hint());
}
