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
#include "arm_compute/runtime/CL/functions/CLGaussianPyramid.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLGaussianPyramidKernel.h"
#include "arm_compute/core/CL/kernels/CLScaleKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "support/ToolchainSupport.h"

#include "arm_compute/runtime/CL/CLPyramid.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLGaussian5x5.h"

#include <cstddef>

using namespace arm_compute;

CLGaussianPyramid::CLGaussianPyramid()
    : _input(nullptr), _pyramid(nullptr), _tmp()
{
}

CLGaussianPyramidHalf::CLGaussianPyramidHalf() // NOLINT
    : _border_handler(),
      _horizontal_reduction(),
      _vertical_reduction()
{
}

void CLGaussianPyramidHalf::configure(ICLTensor *input, CLPyramid *pyramid, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(pyramid == nullptr);
    ARM_COMPUTE_ERROR_ON(input->info()->num_dimensions() != pyramid->get_pyramid_level(0)->info()->num_dimensions());
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) != pyramid->info()->width());
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(1) != pyramid->info()->height());
    ARM_COMPUTE_ERROR_ON(SCALE_PYRAMID_HALF != pyramid->info()->scale());

    /* Get number of pyramid levels */
    const size_t num_levels = pyramid->info()->num_levels();

    _input   = input;
    _pyramid = pyramid;

    if(num_levels > 1)
    {
        _border_handler       = arm_compute::support::cpp14::make_unique<CLFillBorderKernel[]>(num_levels - 1);
        _horizontal_reduction = arm_compute::support::cpp14::make_unique<CLGaussianPyramidHorKernel[]>(num_levels - 1);
        _vertical_reduction   = arm_compute::support::cpp14::make_unique<CLGaussianPyramidVertKernel[]>(num_levels - 1);

        // Apply half scale to the X dimension of the tensor shape
        TensorShape tensor_shape = pyramid->info()->tensor_shape();
        tensor_shape.set(0, (pyramid->info()->width() + 1) * SCALE_PYRAMID_HALF);

        PyramidInfo pyramid_info(num_levels - 1, SCALE_PYRAMID_HALF, tensor_shape, Format::U16);

        _tmp.init(pyramid_info);

        for(size_t i = 0; i < num_levels - 1; ++i)
        {
            /* Configure horizontal kernel */
            _horizontal_reduction[i].configure(_pyramid->get_pyramid_level(i), _tmp.get_pyramid_level(i), border_mode == BorderMode::UNDEFINED);

            /* Configure vertical kernel */
            _vertical_reduction[i].configure(_tmp.get_pyramid_level(i), _pyramid->get_pyramid_level(i + 1), border_mode == BorderMode::UNDEFINED);

            /* Configure border */
            _border_handler[i].configure(_pyramid->get_pyramid_level(i), _horizontal_reduction[i].border_size(), border_mode, PixelValue(constant_border_value));
        }
        _tmp.allocate();
    }
}

void CLGaussianPyramidHalf::run()
{
    ARM_COMPUTE_ERROR_ON_MSG(_pyramid == nullptr, "Unconfigured function");

    /* Get number of pyramid levels */
    const size_t num_levels = _pyramid->info()->num_levels();

    /* The first level of the pyramid has the input image */
    _pyramid->get_pyramid_level(0)->map(CLScheduler::get().queue(), true /* blocking */);
    _input->map(CLScheduler::get().queue(), true /* blocking */);
    _pyramid->get_pyramid_level(0)->copy_from(*_input);
    _input->unmap(CLScheduler::get().queue());
    _pyramid->get_pyramid_level(0)->unmap(CLScheduler::get().queue());

    for(unsigned int i = 0; i < num_levels - 1; ++i)
    {
        CLScheduler::get().enqueue(_border_handler[i], false);
        CLScheduler::get().enqueue(_horizontal_reduction[i], false);
        CLScheduler::get().enqueue(_vertical_reduction[i], false);
    }
}

CLGaussianPyramidOrb::CLGaussianPyramidOrb() // NOLINT
    : _gauss5x5(),
      _scale_nearest()
{
}

void CLGaussianPyramidOrb::configure(ICLTensor *input, CLPyramid *pyramid, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(nullptr == pyramid);
    ARM_COMPUTE_ERROR_ON(input->info()->num_dimensions() != pyramid->get_pyramid_level(0)->info()->num_dimensions());
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) != pyramid->info()->width());
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(1) != pyramid->info()->height());
    ARM_COMPUTE_ERROR_ON(SCALE_PYRAMID_ORB != pyramid->info()->scale());

    /* Get number of pyramid levels */
    const size_t num_levels = pyramid->info()->num_levels();

    _input   = input;
    _pyramid = pyramid;

    if(num_levels > 1)
    {
        _gauss5x5      = arm_compute::support::cpp14::make_unique<CLGaussian5x5[]>(num_levels - 1);
        _scale_nearest = arm_compute::support::cpp14::make_unique<CLScaleKernel[]>(num_levels - 1);

        PyramidInfo pyramid_info(num_levels - 1, SCALE_PYRAMID_ORB, pyramid->info()->tensor_shape(), Format::U8);

        _tmp.init(pyramid_info);

        for(size_t i = 0; i < num_levels - 1; ++i)
        {
            /* Configure gaussian 5x5 */
            _gauss5x5[i].configure(_pyramid->get_pyramid_level(i), _tmp.get_pyramid_level(i), border_mode, constant_border_value);

            /* Configure scale image kernel */
            _scale_nearest[i].configure(_tmp.get_pyramid_level(i), _pyramid->get_pyramid_level(i + 1), InterpolationPolicy::NEAREST_NEIGHBOR, border_mode == BorderMode::UNDEFINED, SamplingPolicy::CENTER);
        }

        _tmp.allocate();
    }
}

void CLGaussianPyramidOrb::run()
{
    ARM_COMPUTE_ERROR_ON_MSG(_pyramid == nullptr, "Unconfigured function");

    /* Get number of pyramid levels */
    const size_t num_levels = _pyramid->info()->num_levels();

    /* The first level of the pyramid has the input image */
    _pyramid->get_pyramid_level(0)->map(CLScheduler::get().queue(), true /* blocking */);
    _input->map(CLScheduler::get().queue(), true /* blocking */);
    _pyramid->get_pyramid_level(0)->copy_from(*_input);
    _input->unmap(CLScheduler::get().queue());
    _pyramid->get_pyramid_level(0)->unmap(CLScheduler::get().queue());

    for(unsigned int i = 0; i < num_levels - 1; ++i)
    {
        _gauss5x5[i].run();
        CLScheduler::get().enqueue(_scale_nearest[i]);
    }
}
