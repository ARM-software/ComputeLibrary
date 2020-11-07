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
#include "arm_compute/runtime/CL/functions/CLCannyEdge.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/functions/CLSobel3x3.h"
#include "arm_compute/runtime/CL/functions/CLSobel5x5.h"
#include "arm_compute/runtime/CL/functions/CLSobel7x7.h"
#include "src/core/CL/kernels/CLCannyEdgeKernel.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/core/CL/kernels/CLSobel5x5Kernel.h"
#include "src/core/CL/kernels/CLSobel7x7Kernel.h"
#include "support/MemorySupport.h"

using namespace arm_compute;

CLCannyEdge::CLCannyEdge(std::shared_ptr<IMemoryManager> memory_manager) // NOLINT
    : _memory_group(std::move(memory_manager)),
      _sobel(),
      _gradient(support::cpp14::make_unique<CLGradientKernel>()),
      _border_mag_gradient(support::cpp14::make_unique<CLFillBorderKernel>()),
      _non_max_suppr(support::cpp14::make_unique<CLEdgeNonMaxSuppressionKernel>()),
      _edge_trace(support::cpp14::make_unique<CLEdgeTraceKernel>()),
      _gx(),
      _gy(),
      _mag(),
      _phase(),
      _nonmax(),
      _visited(),
      _recorded(),
      _l1_list_counter(),
      _l1_stack(),
      _output(nullptr)
{
}

CLCannyEdge::~CLCannyEdge() = default;

void CLCannyEdge::configure(ICLTensor *input, ICLTensor *output, int32_t upper_thr, int32_t lower_thr, int32_t gradient_size, int32_t norm_type, BorderMode border_mode,
                            uint8_t constant_border_value)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, upper_thr, lower_thr, gradient_size, norm_type, border_mode, constant_border_value);
}

void CLCannyEdge::configure(const CLCompileContext &compile_context, ICLTensor *input, ICLTensor *output, int32_t upper_thr, int32_t lower_thr, int32_t gradient_size, int32_t norm_type,
                            BorderMode border_mode,
                            uint8_t    constant_border_value)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON((1 != norm_type) && (2 != norm_type));
    ARM_COMPUTE_ERROR_ON((gradient_size != 3) && (gradient_size != 5) && (gradient_size != 7));
    ARM_COMPUTE_ERROR_ON((lower_thr < 0) || (lower_thr >= upper_thr));

    _output = output;

    const unsigned int L1_hysteresis_stack_size = 8;
    const TensorShape  shape                    = input->info()->tensor_shape();

    TensorInfo gradient_info;
    TensorInfo info;

    // Initialize images
    if(gradient_size < 7)
    {
        gradient_info.init(shape, 1, arm_compute::DataType::S16);
        info.init(shape, 1, arm_compute::DataType::U16);
    }
    else
    {
        gradient_info.init(shape, 1, arm_compute::DataType::S32);
        info.init(shape, 1, arm_compute::DataType::U32);
    }

    _gx.allocator()->init(gradient_info);
    _gy.allocator()->init(gradient_info);
    _mag.allocator()->init(info);
    _nonmax.allocator()->init(info);

    TensorInfo info_u8(shape, 1, arm_compute::DataType::U8);
    _phase.allocator()->init(info_u8);
    _l1_list_counter.allocator()->init(info_u8);

    TensorInfo info_u32(shape, 1, arm_compute::DataType::U32);
    _visited.allocator()->init(info_u32);
    _recorded.allocator()->init(info_u32);

    TensorShape shape_l1_stack = input->info()->tensor_shape();
    shape_l1_stack.set(0, input->info()->dimension(0) * L1_hysteresis_stack_size);
    TensorInfo info_s32(shape_l1_stack, 1, arm_compute::DataType::S32);
    _l1_stack.allocator()->init(info_s32);

    // Manage intermediate buffers
    _memory_group.manage(&_gx);
    _memory_group.manage(&_gy);

    // Configure/Init sobelNxN
    if(gradient_size == 3)
    {
        auto k = arm_compute::support::cpp14::make_unique<CLSobel3x3>();
        k->configure(compile_context, input, &_gx, &_gy, border_mode, constant_border_value);
        _sobel = std::move(k);
    }
    else if(gradient_size == 5)
    {
        auto k = arm_compute::support::cpp14::make_unique<CLSobel5x5>();
        k->configure(compile_context, input, &_gx, &_gy, border_mode, constant_border_value);
        _sobel = std::move(k);
    }
    else if(gradient_size == 7)
    {
        auto k = arm_compute::support::cpp14::make_unique<CLSobel7x7>();
        k->configure(compile_context, input, &_gx, &_gy, border_mode, constant_border_value);
        _sobel = std::move(k);
    }
    else
    {
        ARM_COMPUTE_ERROR_VAR("Gradient size %d not supported", gradient_size);
    }

    // Manage intermediate buffers
    _memory_group.manage(&_mag);
    _memory_group.manage(&_phase);

    // Configure gradient
    _gradient->configure(compile_context, &_gx, &_gy, &_mag, &_phase, norm_type);

    // Allocate intermediate buffers
    _gx.allocator()->allocate();
    _gy.allocator()->allocate();

    // Manage intermediate buffers
    _memory_group.manage(&_nonmax);

    // Configure non-maxima suppression
    _non_max_suppr->configure(compile_context, &_mag, &_phase, &_nonmax, lower_thr, border_mode == BorderMode::UNDEFINED);

    // Allocate intermediate buffers
    _phase.allocator()->allocate();

    // Fill border around magnitude image as non-maxima suppression will access
    // it. If border mode is undefined filling the border is a nop.
    _border_mag_gradient->configure(compile_context, &_mag, _non_max_suppr->border_size(), border_mode, constant_border_value);

    // Allocate intermediate buffers
    _mag.allocator()->allocate();

    // Manage intermediate buffers
    _memory_group.manage(&_visited);
    _memory_group.manage(&_recorded);
    _memory_group.manage(&_l1_stack);
    _memory_group.manage(&_l1_list_counter);

    // Configure edge tracing
    _edge_trace->configure(compile_context, &_nonmax, output, upper_thr, lower_thr, &_visited, &_recorded, &_l1_stack, &_l1_list_counter);

    // Allocate intermediate buffers
    _visited.allocator()->allocate();
    _recorded.allocator()->allocate();
    _l1_stack.allocator()->allocate();
    _l1_list_counter.allocator()->allocate();
    _nonmax.allocator()->allocate();
}

void CLCannyEdge::run()
{
    MemoryGroupResourceScope scope_mg(_memory_group);

    // Run sobel
    _sobel->run();

    // Run phase and magnitude calculation
    CLScheduler::get().enqueue(*_gradient, false);

    // Fill border before non-maxima suppression. Nop for border mode undefined.
    CLScheduler::get().enqueue(*_border_mag_gradient, false);

    // Run non max suppresion
    _nonmax.clear(CLScheduler::get().queue());
    CLScheduler::get().enqueue(*_non_max_suppr, false);

    // Clear temporary structures and run edge trace
    _output->clear(CLScheduler::get().queue());
    _visited.clear(CLScheduler::get().queue());
    _recorded.clear(CLScheduler::get().queue());
    _l1_list_counter.clear(CLScheduler::get().queue());
    _l1_stack.clear(CLScheduler::get().queue());
    CLScheduler::get().enqueue(*_edge_trace, true);
}
