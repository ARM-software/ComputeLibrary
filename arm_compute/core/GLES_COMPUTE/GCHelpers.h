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
#ifndef __ARM_COMPUTE_GCHELPERS_H__
#define __ARM_COMPUTE_GCHELPERS_H__

#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/core/Helpers.h"
#include "support/ToolchainSupport.h"

#include <string>

namespace arm_compute
{
namespace gles_compute
{
/** Helper function to create and return a unique_ptr pointed to a GLES kernel object
 *  It also calls the kernel's configuration.
 *
 * @param[in] args All the arguments that need pass to kernel's configuration.
 *
 * @return A unique pointer pointed to a GLES kernel object
 */
template <typename Kernel, typename... T>
std::unique_ptr<Kernel> create_configure_kernel(T &&... args)
{
    std::unique_ptr<Kernel> k = arm_compute::support::cpp14::make_unique<Kernel>();
    k->configure(std::forward<T>(args)...);
    return k;
}

/** Helper function to create and return a unique_ptr pointed to a GLES kernel object
 *
 * @return A unique pointer pointed to a GLES kernel object
 */
template <typename Kernel>
std::unique_ptr<Kernel> create_kernel()
{
    std::unique_ptr<Kernel> k = arm_compute::support::cpp14::make_unique<Kernel>();
    return k;
}

/** Max vector width of an GLES vector */
static constexpr unsigned int max_gc_vector_width = 16;
} // namespace gles_compute
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GCHELPERS_H__ */
