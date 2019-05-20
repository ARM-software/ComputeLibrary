/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEOpticalFlow.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/kernels/NELKTrackerKernel.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/NEON/functions/NEScharr3x3.h"
#include "arm_compute/runtime/Pyramid.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

NEOpticalFlow::NEOpticalFlow(std::shared_ptr<IMemoryManager> memory_manager) // NOLINT
    : _memory_group(std::move(memory_manager)),
      _func_scharr(),
      _kernel_tracker(),
      _scharr_gx(),
      _scharr_gy(),
      _new_points(nullptr),
      _new_points_estimates(nullptr),
      _old_points(nullptr),
      _new_points_internal(),
      _old_points_internal(),
      _num_levels(0)
{
}

void NEOpticalFlow::configure(const Pyramid *old_pyramid, const Pyramid *new_pyramid, const IKeyPointArray *old_points, const IKeyPointArray *new_points_estimates,
                              IKeyPointArray *new_points, Termination termination, float epsilon, unsigned int num_iterations, size_t window_dimension,
                              bool use_initial_estimate, BorderMode border_mode, uint8_t constant_border_value)
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

    _num_levels           = old_pyramid->info()->num_levels();
    _old_points           = old_points;
    _new_points           = new_points;
    _new_points_estimates = new_points_estimates;

    const float pyr_scale = old_pyramid->info()->scale();

    _func_scharr.clear();
    _kernel_tracker.clear();
    _scharr_gx.clear();
    _scharr_gy.clear();

    _func_scharr.resize(_num_levels);
    _kernel_tracker.resize(_num_levels);
    _scharr_gx.resize(_num_levels);
    _scharr_gy.resize(_num_levels);

    _old_points_internal = LKInternalKeypointArray(old_points->num_values());
    _new_points_internal = LKInternalKeypointArray(old_points->num_values());
    _new_points->resize(old_points->num_values());

    for(unsigned int i = 0; i < _num_levels; ++i)
    {
        // Get images from the ith level of old and right pyramid
        IImage *old_ith_input = old_pyramid->get_pyramid_level(i);
        IImage *new_ith_input = new_pyramid->get_pyramid_level(i);

        // Get width and height of images
        const unsigned int width_ith  = old_ith_input->info()->dimension(0);
        const unsigned int height_ith = new_ith_input->info()->dimension(1);

        TensorInfo tensor_info(TensorShape(width_ith, height_ith), Format::S16);

        _scharr_gx[i].allocator()->init(tensor_info);
        _scharr_gy[i].allocator()->init(tensor_info);

        // Manage intermediate buffers
        _memory_group.manage(&_scharr_gx[i]);
        _memory_group.manage(&_scharr_gy[i]);

        // Init Scharr kernel
        _func_scharr[i].configure(old_ith_input, &_scharr_gx[i], &_scharr_gy[i], border_mode, constant_border_value);

        // Init Lucas-Kanade kernel
        _kernel_tracker[i].configure(old_ith_input, new_ith_input, &_scharr_gx[i], &_scharr_gy[i],
                                     old_points, new_points_estimates, new_points,
                                     &_old_points_internal, &_new_points_internal,
                                     termination, use_initial_estimate, epsilon, num_iterations, window_dimension,
                                     i, _num_levels, pyr_scale);

        _scharr_gx[i].allocator()->allocate();
        _scharr_gy[i].allocator()->allocate();
    }
}

void NEOpticalFlow::run()
{
    ARM_COMPUTE_ERROR_ON_MSG(_num_levels == 0, "Unconfigured function");

    MemoryGroupResourceScope scope_mg(_memory_group);

    for(unsigned int level = _num_levels; level > 0; --level)
    {
        // Run Scharr kernel
        _func_scharr[level - 1].run();

        // Run Lucas-Kanade kernel
        NEScheduler::get().schedule(&_kernel_tracker[level - 1], Window::DimX);
    }
}
