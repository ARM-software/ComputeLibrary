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
#include "arm_compute/runtime/NEON/functions/NECannyEdge.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/kernels/NECannyEdgeKernel.h"
#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/NEON/functions/NESobel3x3.h"
#include "arm_compute/runtime/NEON/functions/NESobel5x5.h"
#include "arm_compute/runtime/NEON/functions/NESobel7x7.h"
#include "arm_compute/runtime/TensorAllocator.h"

#include <cstring>
#include <utility>

using namespace arm_compute;

NECannyEdge::NECannyEdge()
    : _sobel(), _gradient(), _non_max_suppr(), _edge_trace(), _border_mag_gradient(), _border_edge_trace(), _gx(), _gy(), _magnitude(), _phase(), _nonmax(), _output(nullptr)
{
}

void NECannyEdge::configure(ITensor *input, ITensor *output, int32_t upper_thr, int32_t lower_thr, int32_t gradient_size, int32_t norm_type, BorderMode border_mode, uint8_t constant_border_value,
                            bool use_fp16)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(gradient_size < 3);
    ARM_COMPUTE_ERROR_ON(gradient_size > 7);
    ARM_COMPUTE_ERROR_ON(lower_thr > upper_thr);
    ARM_COMPUTE_ERROR_ON((1 != norm_type) && (2 != norm_type));

    _output = output;

    const TensorShape &shape = input->info()->tensor_shape();
    TensorInfo         gradient_info;
    TensorInfo         magnitude_info;

    // Initialize images
    if(gradient_size < 7)
    {
        gradient_info.init(shape, Format::S16);
        magnitude_info.init(shape, Format::U16);
    }
    else
    {
        gradient_info.init(shape, Format::S32);
        magnitude_info.init(shape, Format::U32);
    }

    _gx.allocator()->init(gradient_info);
    _gy.allocator()->init(gradient_info);
    _magnitude.allocator()->init(magnitude_info);

    TensorInfo info(shape, Format::U8);
    _phase.allocator()->init(info);
    _nonmax.allocator()->init(info);

    // Configure/Init sobelNxN
    if(gradient_size == 3)
    {
        auto k = arm_compute::cpp14::make_unique<NESobel3x3>();
        k->configure(input, &_gx, &_gy, border_mode, constant_border_value);
        _sobel = std::move(k);
    }
    else if(gradient_size == 5)
    {
        auto k = arm_compute::cpp14::make_unique<NESobel5x5>();
        k->configure(input, &_gx, &_gy, border_mode, constant_border_value);
        _sobel = std::move(k);
    }
    else if(gradient_size == 7)
    {
        auto k = arm_compute::cpp14::make_unique<NESobel7x7>();
        k->configure(input, &_gx, &_gy, border_mode, constant_border_value);
        _sobel = std::move(k);
    }
    else
    {
        ARM_COMPUTE_ERROR("Gradient size not supported\n");
    }

    // Configure gradient
    if(use_fp16)
    {
        auto k = arm_compute::cpp14::make_unique<NEGradientFP16Kernel>();
        k->configure(&_gx, &_gy, &_magnitude, &_phase, norm_type);
        _gradient = std::move(k);
    }
    else
    {
        auto k = arm_compute::cpp14::make_unique<NEGradientKernel>();
        k->configure(&_gx, &_gy, &_magnitude, &_phase, norm_type);
        _gradient = std::move(k);
    }

    // Configure non-maxima suppression
    _non_max_suppr.configure(&_magnitude, &_phase, &_nonmax, upper_thr, lower_thr, border_mode == BorderMode::UNDEFINED);

    // Fill border around magnitude image as non-maxima suppression will access
    // it. If border mode is undefined filling the border is a nop.
    _border_mag_gradient.configure(&_magnitude, _non_max_suppr.border_size(), border_mode, constant_border_value);

    // Configure edge tracing
    _edge_trace.configure(&_nonmax, output);

    // Fill border with "No edge" to stop recursion in edge trace
    _border_edge_trace.configure(&_nonmax, _edge_trace.border_size(), BorderMode::CONSTANT, 0);

    // Allocate intermediate tensors
    _gx.allocator()->allocate();
    _gy.allocator()->allocate();
    _phase.allocator()->allocate();
    _magnitude.allocator()->allocate();
    _nonmax.allocator()->allocate();
}

void NECannyEdge::run()
{
    ARM_COMPUTE_ERROR_ON_MSG(_sobel == nullptr, "Unconfigured function");
    ARM_COMPUTE_ERROR_ON(_output == nullptr);

    // Run sobelNxN
    _sobel->run();

    // Fill border before non-maxima suppression. Nop for border mode undefined.
    _border_mag_gradient.run(_border_mag_gradient.window());

    // Run gradient
    NEScheduler::get().multithread(_gradient.get());

    // Run non-maxima suppression
    NEScheduler::get().multithread(&_non_max_suppr);

    ARM_COMPUTE_ERROR_ON(_output->buffer() == nullptr);
    memset(_output->buffer(), 0, _output->info()->total_size());

    // Fill border before edge trace
    _border_edge_trace.run(_border_edge_trace.window());

    // Run edge tracing
    _edge_trace.run(_edge_trace.window());
}
