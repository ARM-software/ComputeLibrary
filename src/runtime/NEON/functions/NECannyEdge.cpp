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
#include "arm_compute/runtime/NEON/functions/NECannyEdge.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/NEON/functions/NESobel3x3.h"
#include "arm_compute/runtime/NEON/functions/NESobel5x5.h"
#include "arm_compute/runtime/NEON/functions/NESobel7x7.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "src/core/NEON/kernels/NECannyEdgeKernel.h"
#include "src/core/NEON/kernels/NEFillBorderKernel.h"
#include "src/core/NEON/kernels/NESobel5x5Kernel.h"
#include "src/core/NEON/kernels/NESobel7x7Kernel.h"
#include "support/MemorySupport.h"

#include <cstring>
#include <inttypes.h>
#include <utility>

namespace arm_compute
{
NECannyEdge::~NECannyEdge() = default;

NECannyEdge::NECannyEdge(std::shared_ptr<IMemoryManager> memory_manager) // NOLINT
    : _memory_group(std::move(memory_manager)),
      _sobel(),
      _gradient(),
      _non_max_suppr(),
      _edge_trace(),
      _border_mag_gradient(),
      _border_edge_trace(),
      _gx(),
      _gy(),
      _magnitude(),
      _phase(),
      _nonmax(),
      _output(nullptr)
{
}

void NECannyEdge::configure(ITensor *input, ITensor *output, int32_t upper_thr, int32_t lower_thr, int32_t gradient_size, int32_t norm_type, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON((1 != norm_type) && (2 != norm_type));
    ARM_COMPUTE_ERROR_ON((gradient_size != 3) && (gradient_size != 5) && (gradient_size != 7));
    ARM_COMPUTE_ERROR_ON((lower_thr < 0) || (lower_thr >= upper_thr));

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

    // Manage intermediate buffers
    _memory_group.manage(&_gx);
    _memory_group.manage(&_gy);

    // Configure/Init sobelNxN
    if(gradient_size == 3)
    {
        auto k = arm_compute::support::cpp14::make_unique<NESobel3x3>();
        k->configure(input, &_gx, &_gy, border_mode, constant_border_value);
        _sobel = std::move(k);
    }
    else if(gradient_size == 5)
    {
        auto k = arm_compute::support::cpp14::make_unique<NESobel5x5>();
        k->configure(input, &_gx, &_gy, border_mode, constant_border_value);
        _sobel = std::move(k);
    }
    else if(gradient_size == 7)
    {
        auto k = arm_compute::support::cpp14::make_unique<NESobel7x7>();
        k->configure(input, &_gx, &_gy, border_mode, constant_border_value);
        _sobel = std::move(k);
    }
    else
    {
        ARM_COMPUTE_ERROR_VAR("Gradient size %+" PRId32 " not supported\n", gradient_size);
    }

    // Manage intermediate buffers
    _memory_group.manage(&_magnitude);
    _memory_group.manage(&_phase);

    // Configure gradient
    auto k = arm_compute::support::cpp14::make_unique<NEGradientKernel>();
    k->configure(&_gx, &_gy, &_magnitude, &_phase, norm_type);
    _gradient = std::move(k);

    // Allocate intermediate tensors
    _gx.allocator()->allocate();
    _gy.allocator()->allocate();

    // Manage intermediate buffers
    _memory_group.manage(&_nonmax);

    // Configure non-maxima suppression
    _non_max_suppr = arm_compute::support::cpp14::make_unique<NEEdgeNonMaxSuppressionKernel>();
    _non_max_suppr->configure(&_magnitude, &_phase, &_nonmax, upper_thr, lower_thr, border_mode == BorderMode::UNDEFINED);

    // Fill border around magnitude image as non-maxima suppression will access
    // it. If border mode is undefined filling the border is a nop.
    _border_mag_gradient = arm_compute::support::cpp14::make_unique<NEFillBorderKernel>();
    _border_mag_gradient->configure(&_magnitude, _non_max_suppr->border_size(), border_mode, constant_border_value);

    // Allocate intermediate tensors
    _phase.allocator()->allocate();
    _magnitude.allocator()->allocate();

    // Configure edge tracing
    _edge_trace = arm_compute::support::cpp14::make_unique<NEEdgeTraceKernel>();
    _edge_trace->configure(&_nonmax, output);

    // Fill border with "No edge" to stop recursion in edge trace
    _border_edge_trace = arm_compute::support::cpp14::make_unique<NEFillBorderKernel>();
    _border_edge_trace->configure(&_nonmax, _edge_trace->border_size(), BorderMode::CONSTANT, static_cast<float>(0.f));

    // Allocate intermediate tensors
    _nonmax.allocator()->allocate();
}

void NECannyEdge::run()
{
    ARM_COMPUTE_ERROR_ON_MSG(_sobel == nullptr, "Unconfigured function");

    MemoryGroupResourceScope scope_mg(_memory_group);

    // Run sobelNxN
    _sobel->run();

    // Run gradient
    NEScheduler::get().schedule(_gradient.get(), Window::DimY);

    // Fill border before non-maxima suppression. Nop for border mode undefined.
    NEScheduler::get().schedule(_border_mag_gradient.get(), Window::DimZ);

    // Run non-maxima suppression
    NEScheduler::get().schedule(_non_max_suppr.get(), Window::DimY);

    ARM_COMPUTE_ERROR_ON(_output->buffer() == nullptr);
    std::fill_n(_output->buffer(), _output->info()->total_size(), 0);

    // Fill border before edge trace
    NEScheduler::get().schedule(_border_edge_trace.get(), Window::DimZ);

    // Run edge tracing
    NEScheduler::get().schedule(_edge_trace.get(), Window::DimY);
}
} // namespace arm_compute
