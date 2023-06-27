/*
 * Copyright (c) 2023 Arm Limited.
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

#include "GpuCkwKernelArgumentsHelpers.h"

namespace arm_compute
{
void cl_add_tensor_component_argument(cl::Kernel &kernel, unsigned int &idx, ICLTensor *tensor, ckw::TensorComponentType component)
{
    ARM_COMPUTE_ERROR_ON(tensor == nullptr);

    const auto *info    = tensor->info();
    const auto &strides = info->strides_in_bytes();

    switch(component)
    {
        case ckw::TensorComponentType::OffsetFirstElement:
            kernel.setArg<cl_uint>(idx++, info->offset_first_element_in_bytes());
            break;
        case ckw::TensorComponentType::Stride0:
            kernel.setArg<cl_uint>(idx++, strides[0]);
            break;
        case ckw::TensorComponentType::Stride1:
            kernel.setArg<cl_uint>(idx++, strides[1]);
            break;
        case ckw::TensorComponentType::Stride2:
            kernel.setArg<cl_uint>(idx++, strides[2]);
            break;
        case ckw::TensorComponentType::Stride3:
            kernel.setArg<cl_uint>(idx++, strides[3]);
            break;
        case ckw::TensorComponentType::Stride4:
            kernel.setArg<cl_uint>(idx++, strides[4]);
            break;
        case ckw::TensorComponentType::Dim0:
            kernel.setArg<cl_uint>(idx++, info->dimension(0));
            break;
        case ckw::TensorComponentType::Dim1:
            kernel.setArg<cl_uint>(idx++, info->dimension(1));
            break;
        case ckw::TensorComponentType::Dim2:
            kernel.setArg<cl_uint>(idx++, info->dimension(2));
            break;
        case ckw::TensorComponentType::Dim3:
            kernel.setArg<cl_uint>(idx++, info->dimension(3));
            break;
        case ckw::TensorComponentType::Dim4:
            kernel.setArg<cl_uint>(idx++, info->dimension(4));
            break;
        case ckw::TensorComponentType::Dim1xDim2:
            kernel.setArg<cl_uint>(idx++, info->dimension(1) * info->dimension(2));
            break;
        case ckw::TensorComponentType::Dim2xDim3:
            kernel.setArg<cl_uint>(idx++, info->dimension(2) * info->dimension(3));
            break;
        case ckw::TensorComponentType::Dim1xDim2xDim3:
            kernel.setArg<cl_uint>(idx++, info->dimension(1) * info->dimension(2) * info->dimension(3));
            break;
        case ckw::TensorComponentType::Unknown:
        default:
            ARM_COMPUTE_ERROR("Unknown tensor component");
    }
}

void cl_add_buffer_argument(cl::Kernel &kernel, unsigned int &idx, const cl::Buffer &buffer)
{
    kernel.setArg(idx++, buffer);
}

void cl_add_texture_argument(cl::Kernel &kernel, unsigned int &idx, const cl::Image &image)
{
    kernel.setArg(idx++, image);
}

} // namespace arm_compute
