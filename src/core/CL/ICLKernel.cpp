/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#include "arm_compute/core/CL/ICLKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <cstddef>

using namespace arm_compute;

void arm_compute::enqueue(cl::CommandQueue &queue, ICLKernel &kernel, const Window &window, const cl::NDRange &lws_hint, bool use_dummy_work_items)
{
    if(kernel.kernel()() == nullptr)
    {
        return;
    }

    for(unsigned int i = 0; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_ERROR_ON(window[i].step() == 0);
        // Make sure that dimensions > Z are 1
        ARM_COMPUTE_ERROR_ON((i >= 3) && ((window[i].end() - window[i].start()) != 1));
    }

    cl::NDRange gws = ICLKernel::gws_from_window(window);

    // Check for empty NDRange
    if(gws.dimensions() == 0)
    {
        return;
    }

    // Use dummy work-items
    if(use_dummy_work_items)
    {
        gws.get()[0] = get_next_power_two(gws[0]);
        gws.get()[1] = get_next_power_two(gws[1]);
    }

    cl::NDRange valid_lws;
    if(lws_hint[0] * lws_hint[1] * lws_hint[2] > kernel.get_max_workgroup_size())
    {
        valid_lws = cl::NullRange;
    }
    else
    {
        valid_lws = lws_hint;
    }

    cl::NDRange lws = cl::NullRange;

    if((valid_lws[0] <= gws[0]) && (valid_lws[1] <= gws[1]) && (valid_lws[2] <= gws[2]))
    {
        lws = valid_lws;
    }

    queue.enqueueNDRangeKernel(kernel.kernel(), cl::NullRange, gws, lws);
}

template <unsigned int dimension_size>
void ICLKernel::add_tensor_argument(unsigned &idx, const ICLTensor *tensor, const Window &window)
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
    _kernel.setArg(idx++, tensor->cl_buffer());

    for(unsigned int d = 0; d < dimension_size; ++d)
    {
        _kernel.setArg<cl_uint>(idx++, strides[d]);
        _kernel.setArg<cl_uint>(idx++, strides[d] * window[d].step());
    }

    _kernel.setArg<cl_uint>(idx++, offset_first_element);

    ARM_COMPUTE_ERROR_ON_MSG(idx_start + num_arguments_per_tensor<dimension_size>() != idx,
                             "add_%dD_tensor_argument() is supposed to add exactly %d arguments to the kernel", dimension_size, num_arguments_per_tensor<dimension_size>());
    ARM_COMPUTE_UNUSED(idx_start);
}

#ifndef DOXYGEN_SKIP_THIS
template void ICLKernel::add_tensor_argument<1>(unsigned &idx, const ICLTensor *tensor, const Window &window);
template void ICLKernel::add_tensor_argument<2>(unsigned &idx, const ICLTensor *tensor, const Window &window);
template void ICLKernel::add_tensor_argument<3>(unsigned &idx, const ICLTensor *tensor, const Window &window);
template void ICLKernel::add_tensor_argument<4>(unsigned &idx, const ICLTensor *tensor, const Window &window);
#endif /* DOXYGEN_SKIP_THIS */

void ICLKernel::set_target(cl::Device &device)
{
    _target = get_target_from_device(device);
}

size_t ICLKernel::get_max_workgroup_size()
{
    if(_max_workgroup_size == 0)
    {
        _max_workgroup_size = CLKernelLibrary::get().max_local_workgroup_size(_kernel);
    }
    return _max_workgroup_size;
}

cl::NDRange ICLKernel::gws_from_window(const Window &window)
{
    if((window.x().end() - window.x().start()) == 0 || (window.y().end() - window.y().start()) == 0)
    {
        return cl::NullRange;
    }

    cl::NDRange gws((window.x().end() - window.x().start()) / window.x().step(),
                    (window.y().end() - window.y().start()) / window.y().step(),
                    (window.z().end() - window.z().start()) / window.z().step());

    return gws;
}
