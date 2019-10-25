/*
 * Copyright (c) 2019 ARM Limited.
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
#include "tests/framework/ParametersLibrary.h"

namespace arm_compute
{
namespace test
{
void ParametersLibrary::set_cpu_ctx(std::unique_ptr<IRuntimeContext> cpu_ctx)
{
    _cpu_ctx = std::move(cpu_ctx);
}

void ParametersLibrary::set_cl_ctx(std::unique_ptr<IRuntimeContext> cl_ctx)
{
    _cl_ctx = std::move(cl_ctx);
}

void ParametersLibrary::set_gc_ctx(std::unique_ptr<IRuntimeContext> gc_ctx)
{
    _gc_ctx = std::move(gc_ctx);
}

template <>
typename ContextType<Tensor>::type *ParametersLibrary::get_ctx<Tensor>()
{
    return _cpu_ctx.get();
}

#if ARM_COMPUTE_CL
template <>
typename ContextType<CLTensor>::type *ParametersLibrary::get_ctx<CLTensor>()
{
    return static_cast<typename ContextType<CLTensor>::type *>(_cl_ctx.get());
}
#endif /* ARM_COMPUTE_CL */

#if ARM_COMPUTE_GC
template <>
typename ContextType<GCTensor>::type *ParametersLibrary::get_ctx<GCTensor>()
{
    return static_cast<typename ContextType<GCTensor>::type *>(_gc_ctx.get());
}
#endif /* ARM_COMPUTE_GC */
} // namespace test
} // namespace arm_compute
