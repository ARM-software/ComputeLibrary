/*
 * Copyright (c) 2020 Arm Limited.
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
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Types.h"

#include "src/core/CL/CLUtils.h"

cl::Image2D arm_compute::create_image2d_from_buffer(const cl::Context &ctx, const cl::Buffer &buffer, const TensorShape &shape2d, cl_channel_type data_type, size_t image_row_pitch)
{
    cl_mem cl_image;
    cl_int err = CL_SUCCESS;

    const cl_image_format format = { CL_RGBA, data_type };

    cl_image_desc desc;
    memset(&desc, 0, sizeof(desc));
    desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
    desc.mem_object      = buffer();
    desc.image_row_pitch = image_row_pitch;
    desc.image_width     = shape2d[0];
    desc.image_height    = shape2d[1];

    cl_image = clCreateImage(ctx(), CL_MEM_READ_ONLY, &format, &desc, nullptr, &err);

    ARM_COMPUTE_UNUSED(err);
    ARM_COMPUTE_ERROR_ON_MSG(err != CL_SUCCESS, "Error during the creation of CL image from buffer");

    return cl::Image2D(cl_image);
}
