/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/core/Error.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/runtime/CL/CLPyramid.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLGaussian5x5.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/core/CL/kernels/CLGaussian5x5Kernel.h"
#include "src/core/CL/kernels/CLGaussianPyramidKernel.h"
#include "src/core/CL/kernels/CLScaleKernel.h"
#include "support/MemorySupport.h"

#include <cstddef>

using namespace arm_compute;

CLGaussianPyramid::CLGaussianPyramid()
    : _input(nullptr), _pyramid(nullptr), _tmp()
{
}

CLGaussianPyramid::~CLGaussianPyramid() = default;

CLGaussianPyramidHalf::CLGaussianPyramidHalf() // NOLINT
    : _horizontal_border_handler(),
      _vertical_border_handler(),
      _horizontal_reduction(),
      _vertical_reduction()
{
}

CLGaussianPyramidHalf::~CLGaussianPyramidHalf() = default;

void CLGaussianPyramidHalf::configure(ICLTensor *input, CLPyramid *pyramid, BorderMode border_mode, uint8_t constant_border_value)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, pyramid, border_mode, constant_border_value);
}

void CLGaussianPyramidHalf::configure(const CLCompileContext &compile_context, ICLTensor *input, CLPyramid *pyramid, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(pyramid == nullptr);
    ARM_COMPUTE_ERROR_ON(input->info()->num_dimensions() != pyramid->get_pyramid_level(0)->info()->num_dimensions());
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) != pyramid->info()->width());
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(1) != pyramid->info()->height());
    ARM_COMPUTE_ERROR_ON(SCALE_PYRAMID_HALF != pyramid->info()->scale());

    // Constant value to use for vertical fill border when the border mode is CONSTANT
    const uint16_t pixel_value_u16 = static_cast<uint16_t>(constant_border_value) * 2 + static_cast<uint16_t>(constant_border_value) * 8 + static_cast<uint16_t>(constant_border_value) * 6;

    /* Get number of pyramid levels */
    const size_t num_levels = pyramid->info()->num_levels();

    _input   = input;
    _pyramid = pyramid;

    if(num_levels > 1)
    {
        _horizontal_border_handler.reserve(num_levels - 1);
        _vertical_border_handler.reserve(num_levels - 1);
        _horizontal_reduction.reserve(num_levels - 1);
        _vertical_reduction.reserve(num_levels - 1);

        // Apply half scale to the X dimension of the tensor shape
        TensorShape tensor_shape = pyramid->info()->tensor_shape();
        tensor_shape.set(0, (pyramid->info()->width() + 1) * SCALE_PYRAMID_HALF);

        PyramidInfo pyramid_info(num_levels - 1, SCALE_PYRAMID_HALF, tensor_shape, Format::U16);
        _tmp.init(pyramid_info);

        for(size_t i = 0; i < num_levels - 1; ++i)
        {
            /* Configure horizontal kernel */
            _horizontal_reduction.emplace_back(support::cpp14::make_unique<CLGaussianPyramidHorKernel>());
            _horizontal_reduction.back()->configure(compile_context, _pyramid->get_pyramid_level(i), _tmp.get_pyramid_level(i));

            /* Configure vertical kernel */
            _vertical_reduction.emplace_back(support::cpp14::make_unique<CLGaussianPyramidVertKernel>());
            _vertical_reduction.back()->configure(compile_context, _tmp.get_pyramid_level(i), _pyramid->get_pyramid_level(i + 1));

            /* Configure border */
            _horizontal_border_handler.emplace_back(support::cpp14::make_unique<CLFillBorderKernel>());
            _horizontal_border_handler.back()->configure(compile_context, _pyramid->get_pyramid_level(i), _horizontal_reduction.back()->border_size(), border_mode, PixelValue(constant_border_value));

            /* Configure border */
            _vertical_border_handler.emplace_back(support::cpp14::make_unique<CLFillBorderKernel>());
            _vertical_border_handler.back()->configure(compile_context, _tmp.get_pyramid_level(i), _vertical_reduction.back()->border_size(), border_mode, PixelValue(pixel_value_u16));
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
        CLScheduler::get().enqueue(*_horizontal_border_handler[i], false);
        CLScheduler::get().enqueue(*_horizontal_reduction[i], false);
        CLScheduler::get().enqueue(*_vertical_border_handler[i], false);
        CLScheduler::get().enqueue(*_vertical_reduction[i], false);
    }
}

CLGaussianPyramidOrb::CLGaussianPyramidOrb() // NOLINT
    : _gauss5x5(),
      _scale_nearest()
{
}

void CLGaussianPyramidOrb::configure(ICLTensor *input, CLPyramid *pyramid, BorderMode border_mode, uint8_t constant_border_value)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, pyramid, border_mode, constant_border_value);
}

void CLGaussianPyramidOrb::configure(const CLCompileContext &compile_context, ICLTensor *input, CLPyramid *pyramid, BorderMode border_mode, uint8_t constant_border_value)
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
        _gauss5x5.resize(num_levels - 1);
        _scale_nearest.reserve(num_levels - 1);

        PyramidInfo pyramid_info(num_levels - 1, SCALE_PYRAMID_ORB, pyramid->info()->tensor_shape(), Format::U8);

        _tmp.init(pyramid_info);

        for(size_t i = 0; i < num_levels - 1; ++i)
        {
            /* Configure gaussian 5x5 */
            _gauss5x5[i].configure(compile_context, _pyramid->get_pyramid_level(i), _tmp.get_pyramid_level(i), border_mode, constant_border_value);

            /* Configure scale image kernel */
            _scale_nearest.emplace_back(support::cpp14::make_unique<CLScaleKernel>());
            _scale_nearest.back()->configure(compile_context, _tmp.get_pyramid_level(i), _pyramid->get_pyramid_level(i + 1), ScaleKernelInfo{ InterpolationPolicy::NEAREST_NEIGHBOR, border_mode, PixelValue(), SamplingPolicy::CENTER });
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
        CLScheduler::get().enqueue(*_scale_nearest[i]);
    }
}
