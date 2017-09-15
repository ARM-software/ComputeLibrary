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
#include "arm_compute/runtime/CL/functions/CLOpticalFlow.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLLKTrackerKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/runtime/CL/CLPyramid.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLScharr3x3.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

CLOpticalFlow::CLOpticalFlow(std::shared_ptr<IMemoryManager> memory_manager) // NOLINT
    : _memory_group(std::move(memory_manager)),
      _tracker_init_kernel(),
      _tracker_stage0_kernel(),
      _tracker_stage1_kernel(),
      _tracker_finalize_kernel(),
      _func_scharr(),
      _scharr_gx(),
      _scharr_gy(),
      _old_points(nullptr),
      _new_points_estimates(nullptr),
      _new_points(nullptr),
      _old_points_internal(),
      _new_points_internal(),
      _coefficient_table(),
      _old_values(),
      _num_levels(0)
{
}

void CLOpticalFlow::configure(const CLPyramid *old_pyramid, const CLPyramid *new_pyramid,
                              const ICLKeyPointArray *old_points, const ICLKeyPointArray *new_points_estimates, ICLKeyPointArray *new_points,
                              Termination termination, float epsilon, size_t num_iterations, size_t window_dimension, bool use_initial_estimate,
                              BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON(nullptr == old_pyramid);
    ARM_COMPUTE_ERROR_ON(nullptr == new_pyramid);
    ARM_COMPUTE_ERROR_ON(nullptr == old_points);
    ARM_COMPUTE_ERROR_ON(nullptr == new_points_estimates);
    ARM_COMPUTE_ERROR_ON(nullptr == new_points);
    ARM_COMPUTE_ERROR_ON(old_pyramid->info()->num_levels() != new_pyramid->info()->num_levels());
    ARM_COMPUTE_ERROR_ON(0 == old_pyramid->info()->num_levels());
    ARM_COMPUTE_ERROR_ON(old_pyramid->info()->width() != new_pyramid->info()->width());
    ARM_COMPUTE_ERROR_ON(old_pyramid->info()->height() != new_pyramid->info()->height());
    ARM_COMPUTE_ERROR_ON(use_initial_estimate && old_points->num_values() != new_points_estimates->num_values());

    // Set member variables
    _old_points           = old_points;
    _new_points_estimates = new_points_estimates;
    _new_points           = new_points;
    _num_levels           = old_pyramid->info()->num_levels();

    const float pyr_scale              = old_pyramid->info()->scale();
    const int   list_length            = old_points->num_values();
    const int   old_values_list_length = list_length * window_dimension * window_dimension;

    // Create kernels and tensors
    _tracker_init_kernel   = arm_compute::support::cpp14::make_unique<CLLKTrackerInitKernel[]>(_num_levels);
    _tracker_stage0_kernel = arm_compute::support::cpp14::make_unique<CLLKTrackerStage0Kernel[]>(_num_levels);
    _tracker_stage1_kernel = arm_compute::support::cpp14::make_unique<CLLKTrackerStage1Kernel[]>(_num_levels);
    _func_scharr           = arm_compute::support::cpp14::make_unique<CLScharr3x3[]>(_num_levels);
    _scharr_gx             = arm_compute::support::cpp14::make_unique<CLTensor[]>(_num_levels);
    _scharr_gy             = arm_compute::support::cpp14::make_unique<CLTensor[]>(_num_levels);

    // Create internal keypoint arrays
    _old_points_internal = arm_compute::support::cpp14::make_unique<CLLKInternalKeypointArray>(list_length);
    _old_points_internal->resize(list_length);
    _new_points_internal = arm_compute::support::cpp14::make_unique<CLLKInternalKeypointArray>(list_length);
    _new_points_internal->resize(list_length);
    _coefficient_table = arm_compute::support::cpp14::make_unique<CLCoefficientTableArray>(list_length);
    _coefficient_table->resize(list_length);
    _old_values = arm_compute::support::cpp14::make_unique<CLOldValueArray>(old_values_list_length);
    _old_values->resize(old_values_list_length);
    _new_points->resize(list_length);

    for(size_t i = 0; i < _num_levels; ++i)
    {
        // Get images from the ith level of old and right pyramid
        ICLImage *old_ith_input = old_pyramid->get_pyramid_level(i);
        ICLImage *new_ith_input = new_pyramid->get_pyramid_level(i);

        // Get width and height of images
        const unsigned int width_ith  = old_ith_input->info()->dimension(0);
        const unsigned int height_ith = new_ith_input->info()->dimension(1);

        // Initialize Scharr tensors
        TensorInfo tensor_info(TensorShape(width_ith, height_ith), 1, DataType::S16);
        _scharr_gx[i].allocator()->init(tensor_info);
        _scharr_gy[i].allocator()->init(tensor_info);

        // Manage intermediate buffers
        _memory_group.manage(_scharr_gx.get() + i);
        _memory_group.manage(_scharr_gy.get() + i);

        // Init Scharr kernel
        _func_scharr[i].configure(old_ith_input, &_scharr_gx[i], &_scharr_gy[i], border_mode, constant_border_value);

        // Init Lucas-Kanade init kernel
        _tracker_init_kernel[i].configure(old_points, new_points_estimates, _old_points_internal.get(), _new_points_internal.get(), use_initial_estimate, i, _num_levels, pyr_scale);

        // Init Lucas-Kanade stage0 kernel
        _tracker_stage0_kernel[i].configure(old_ith_input, &_scharr_gx[i], &_scharr_gy[i],
                                            _old_points_internal.get(), _new_points_internal.get(), _coefficient_table.get(), _old_values.get(),
                                            window_dimension, i);

        // Init Lucas-Kanade stage1 kernel
        _tracker_stage1_kernel[i].configure(new_ith_input, _new_points_internal.get(), _coefficient_table.get(), _old_values.get(),
                                            termination, epsilon, num_iterations, window_dimension, i);

        // Allocate intermediate buffers
        _scharr_gx[i].allocator()->allocate();
        _scharr_gy[i].allocator()->allocate();
    }

    // Finalize Lucas-Kanade
    _tracker_finalize_kernel.configure(_new_points_internal.get(), new_points);
}

void CLOpticalFlow::run()
{
    ARM_COMPUTE_ERROR_ON_MSG(_num_levels == 0, "Unconfigured function");

    _memory_group.acquire();

    for(unsigned int level = _num_levels; level > 0; --level)
    {
        // Run Scharr kernel
        _func_scharr[level - 1].run();

        // Run Lucas-Kanade init kernel
        CLScheduler::get().enqueue(_tracker_init_kernel[level - 1]);

        // Run Lucas-Kanade stage0 kernel
        CLScheduler::get().enqueue(_tracker_stage0_kernel[level - 1]);

        // Run Lucas-Kanade stage1 kernel
        CLScheduler::get().enqueue(_tracker_stage1_kernel[level - 1]);
    }

    CLScheduler::get().enqueue(_tracker_finalize_kernel, true);

    _memory_group.release();
}
