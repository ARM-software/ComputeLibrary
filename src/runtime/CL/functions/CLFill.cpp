/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLFill.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CL/ICLKernel.h"
#include "src/gpu/cl/operators/ClFill.h"

#include <utility>

namespace arm_compute
{
struct CLFill::Impl
{
    const ICLTensor                *src{ nullptr };
    ICLTensor                      *dst{ nullptr };
    std::unique_ptr<opencl::ClFill> op{ nullptr };
};

CLFill::CLFill()
    : _impl(std::make_unique<Impl>())
{
}
CLFill::CLFill(CLFill &&) = default;
CLFill &CLFill::operator=(CLFill &&) = default;
CLFill::~CLFill()                    = default;

void CLFill::configure(ICLTensor *tensor, const PixelValue &constant_value, Window *dst_window)
{
    configure(CLKernelLibrary::get().get_compile_context(), tensor, constant_value, dst_window);
}

void CLFill::configure(const CLCompileContext &compile_context, ICLTensor *tensor, const PixelValue &constant_value, Window *dst_window)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(tensor);

    _impl->src = tensor;

    _impl->op = std::make_unique<opencl::ClFill>();
    _impl->op->configure(compile_context, _impl->src->info(), constant_value, dst_window);
}

Status CLFill::validate(const ITensorInfo *tensor, const PixelValue &constant_value, Window *dst_window)
{
    return opencl::ClFill::validate(tensor, constant_value, dst_window);
}

void CLFill::run()
{
    ITensorPack pack;
    pack.add_tensor(TensorType::ACL_SRC, _impl->src);
    _impl->op->run(pack);
}
} // namespace arm_compute
