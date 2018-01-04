/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCScheduler.h"

#include <cstddef>
#include <sstream>

using namespace arm_compute;

void arm_compute::enqueue(IGCKernel &kernel, const Window &window, const gles::NDRange &lws)
{
    ARM_COMPUTE_UNUSED(kernel);

    if(kernel.kernel().get_program() == 0)
    {
        return;
    }

    ARM_COMPUTE_ERROR_ON((0 == (window.x().end() - window.x().start())) || (0 == (window.y().end() - window.y().start())));

    ARM_COMPUTE_ERROR_ON_MSG((((window.x().end() - window.x().start()) % (window.x().step() * lws[0])) != 0),
                             "window x end =%d, start=%d, step=%d, lws x=%d", window.x().end(), window.x().start(), window.x().step(), lws[0]);
    ARM_COMPUTE_ERROR_ON_MSG((((window.y().end() - window.y().start()) % (window.y().step() * lws[1])) != 0),
                             "window y end =%d, start=%d, step=%d, lws y=%d", window.y().end(), window.y().start(), window.y().step(), lws[1]);
    ARM_COMPUTE_ERROR_ON_MSG((((window.z().end() - window.z().start()) % (window.z().step() * lws[2])) != 0),
                             "window z end =%d, start=%d, step=%d, lws z=%d", window.z().end(), window.z().start(), window.z().step(), lws[2]);

    ARM_COMPUTE_GL_CHECK(glDispatchCompute(((window.x().end() - window.x().start()) / window.x().step()) / lws[0],
                                           ((window.y().end() - window.y().start()) / window.y().step()) / lws[1],
                                           ((window.z().end() - window.z().start()) / window.z().step()) / lws[2]));
}

IGCKernel::IGCKernel()
    : _kernel(), _lws_hint(gles::NDRange(1U, 1U, 1U))
{
}

GCKernel &IGCKernel::kernel()
{
    return _kernel;
}

template <unsigned int dimension_size>
unsigned int           IGCKernel::num_arguments_per_tensor() const
{
    // Rounding up the tensor attributes structure in compute shader to a multiple of a vec4
    return ceil_to_multiple(1 + 2 * dimension_size, 4);
}

template <unsigned int dimension_size>
void IGCKernel::add_tensor_argument(unsigned int &idx, const IGCTensor *tensor, const unsigned int binding_point, const Window &window)
{
    ARM_COMPUTE_ERROR_ON(tensor == nullptr);

    const ITensorInfo *info    = tensor->info();
    const Strides     &strides = info->strides_in_bytes();

    // Calculate offset to the start of the window
    unsigned int offset_first_element = info->offset_first_element_in_bytes();

    for(unsigned int n = 0; n < info->num_dimensions(); ++n)
    {
        offset_first_element += window[n].start() * strides[n];
    }

    unsigned int idx_start = idx;

    for(unsigned int dimension = 0; dimension < dimension_size; dimension++)
    {
        _kernel.set_argument(idx++, strides[dimension]);
        _kernel.set_argument(idx++, strides[dimension] * window[dimension].step());
    }

    _kernel.set_argument(idx++, offset_first_element);

    // Rounding up the tensor attributes structure in compute shader to a multiple of a vec4
    unsigned int idx_end = ceil_to_multiple(idx, 4);
    for(unsigned int i = idx; i < idx_end; ++i)
    {
        _kernel.set_argument(i, 0);
    }
    idx = idx_end;

    ARM_COMPUTE_GL_CHECK(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point, tensor->gc_buffer()));

    ARM_COMPUTE_ERROR_ON_MSG(idx_start + num_arguments_per_tensor<dimension_size>() != idx,
                             "add_%dD_tensor_argument() is supposed to add exactly %d arguments to the kernel", dimension_size, num_arguments_per_tensor<dimension_size>());
    ARM_COMPUTE_UNUSED(idx_start);
}

void IGCKernel::add_1D_tensor_argument(unsigned int &idx, const IGCTensor *tensor, const unsigned int binding_point, const Window &window)
{
    add_tensor_argument<1>(idx, tensor, binding_point, window);
}

void IGCKernel::add_2D_tensor_argument(unsigned int &idx, const IGCTensor *tensor, const unsigned int binding_point, const Window &window)
{
    add_tensor_argument<2>(idx, tensor, binding_point, window);
}

void IGCKernel::add_3D_tensor_argument(unsigned int &idx, const IGCTensor *tensor, const unsigned int binding_point, const Window &window)
{
    add_tensor_argument<3>(idx, tensor, binding_point, window);
}

unsigned int IGCKernel::num_arguments_per_1D_tensor() const
{
    return num_arguments_per_tensor<1>();
}

unsigned int IGCKernel::num_arguments_per_2D_tensor() const
{
    return num_arguments_per_tensor<2>();
}

unsigned int IGCKernel::num_arguments_per_3D_tensor() const
{
    return num_arguments_per_tensor<3>();
}
