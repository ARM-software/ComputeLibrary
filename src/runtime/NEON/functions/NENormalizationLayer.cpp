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

#include "arm_compute/runtime/NEON/functions/NENormalizationLayer.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/core/NEON/kernels/NENormalizationLayerKernel.h"
#include "support/MemorySupport.h"

namespace arm_compute
{
NENormalizationLayer::~NENormalizationLayer() = default;

NENormalizationLayer::NENormalizationLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _norm_kernel(), _multiply_f(), _input_squared()
{
}

void NENormalizationLayer::configure(const ITensor *input, ITensor *output, const NormalizationLayerInfo &norm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    TensorInfo tensor_info(input->info()->tensor_shape(), 1, input->info()->data_type());
    _input_squared.allocator()->init(tensor_info);

    // Manage intermediate buffers
    _memory_group.manage(&_input_squared);

    // Configure kernels
    _norm_kernel = arm_compute::support::cpp14::make_unique<NENormalizationLayerKernel>();
    _norm_kernel->configure(input, &_input_squared, output, norm_info);
    _multiply_f.configure(input, input, &_input_squared, 1.0f, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO);

    // Allocate the tensor once the configure methods have been called
    _input_squared.allocator()->allocate();
}

Status NENormalizationLayer::validate(const ITensorInfo *input, const ITensorInfo *output, const NormalizationLayerInfo &norm_info)
{
    // Perform validation step
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);

    ARM_COMPUTE_RETURN_ON_ERROR(NENormalizationLayerKernel::validate(input, input, output, norm_info));
    ARM_COMPUTE_RETURN_ON_ERROR(NEPixelWiseMultiplication::validate(input, input, output, 1.0f, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO));

    return Status{};
}

void NENormalizationLayer::run()
{
    MemoryGroupResourceScope scope_mg(_memory_group);
    _multiply_f.run();
    NEScheduler::get().schedule(_norm_kernel.get(), Window::DimY);
}
}