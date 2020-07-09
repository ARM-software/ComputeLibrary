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
#ifndef ARM_COMPUTE_GCHELPERS_H
#define ARM_COMPUTE_GCHELPERS_H

#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"
#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/Helpers.h"

#include <set>
#include <string>

namespace arm_compute
{
// Forward declarations
class GCCoreRuntimeContext;

/** Max vector width of an GLES vector */
static constexpr unsigned int max_gc_vector_width = 16;

/** Helper function to get the GPU target from GLES using GL_RENDERER enum
 *
 * @return the GPU target
 */
GPUTarget get_target_from_device();
/** Creates an GLES kernel
 *
 * @param[in] ctx         A context to be used to create the GLES kernel.
 * @param[in] kernel_name The kernel name.
 * @param[in] build_opts  The build options to be used for the GLES kernel compilation.
 *
 * @return A GLES kernel
 */
GCKernel create_opengl_kernel(GCCoreRuntimeContext *ctx, const std::string &kernel_name, const std::set<std::string> &build_opts);
} // namespace arm_compute
#endif /* ARM_COMPUTE_GCHELPERS_H */
