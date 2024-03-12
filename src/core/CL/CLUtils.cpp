/*
 * Copyright (c) 2020-2023 Arm Limited.
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
#include "src/core/CL/CLUtils.h"

#include "arm_compute/core/CL/CLCompileContext.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/utils/ActivationFunctionUtils.h"
#include "arm_compute/core/utils/StringUtils.h"
#include "arm_compute/core/Validate.h"

#include "support/StringSupport.h"

namespace arm_compute
{
cl::Image2D create_image2d_from_tensor(const ICLTensor *tensor, CLImage2DType image_type)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(tensor);

    const cl::Context &ctx    = CLKernelLibrary::get().context();
    const cl::Buffer  &buffer = tensor->cl_buffer();
    const ITensorInfo *info   = tensor->info();
    ARM_COMPUTE_ERROR_ON_MSG(info->lock_paddings(), "Tensor paddings must not be locked to allow extending paddings to "
                                                    "satisfy cl_image pitch alignment requirement");

    const size_t image_w{info->dimension(0) / 4};
    const size_t image_h{info->tensor_shape().total_size() / info->dimension(0)};
    const size_t max_image_w{CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>()};
    const size_t max_image_h{CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>()};

    ARM_COMPUTE_UNUSED(max_image_w, max_image_h);
    ARM_COMPUTE_ERROR_ON_MSG(image_w > max_image_w, "Image width exceeds maximum width for exporting to cl_image");
    ARM_COMPUTE_ERROR_ON_MSG(image_h > max_image_h, "Image height exceeds maximum height for exporting to cl_image");

    const TensorShape shape2d(image_w, image_h);
    const size_t      image_row_pitch = info->strides_in_bytes()[1];

    return create_image2d_from_buffer(ctx, buffer, shape2d, info->data_type(), image_row_pitch, image_type);
}

cl::Image2D create_image2d_from_buffer(const cl::Context &ctx,
                                       const cl::Buffer  &buffer,
                                       const TensorShape &shape2d,
                                       DataType           data_type,
                                       size_t             image_row_pitch,
                                       CLImage2DType      image_type)
{
    ARM_COMPUTE_ERROR_ON_MSG(!image2d_from_buffer_supported(CLKernelLibrary::get().get_device()),
                             "The extension cl_khr_image2d_from_buffer is not supported on the target platform");
    ARM_COMPUTE_ERROR_ON_MSG(get_cl_image_pitch_alignment(CLKernelLibrary::get().get_device()) == 0,
                             "Impossible to retrieve the cl_image pitch alignment");
    ARM_COMPUTE_ERROR_ON_MSG(buffer.get() == nullptr, "Cannot create cl_image from empty cl_buffer");

    cl_channel_type cl_data_type;

    switch (data_type)
    {
        case DataType::F32:
            cl_data_type = CL_FLOAT;
            break;
        case DataType::F16:
            cl_data_type = CL_HALF_FLOAT;
            break;
        default:
            ARM_COMPUTE_ERROR("Data type not support with OpenCL image2d");
    }

    cl_mem cl_image;
    cl_int err = CL_SUCCESS;

    const cl_image_format format = {CL_RGBA, cl_data_type};

    cl_image_desc desc;
    memset(&desc, 0, sizeof(desc));
    desc.image_type      = CL_MEM_OBJECT_IMAGE2D;
    desc.mem_object      = buffer();
    desc.image_row_pitch = image_row_pitch;
    desc.image_width     = shape2d[0];
    desc.image_height    = shape2d[1];

    switch (image_type)
    {
        case CLImage2DType::ReadOnly:
            cl_image = clCreateImage(ctx(), CL_MEM_READ_ONLY, &format, &desc, nullptr, &err);
            break;
        case CLImage2DType::WriteOnly:
            cl_image = clCreateImage(ctx(), CL_MEM_WRITE_ONLY, &format, &desc, nullptr, &err);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported CLImage2DType");
    }

    ARM_COMPUTE_UNUSED(err);
    ARM_COMPUTE_ERROR_ON_MSG(err != CL_SUCCESS, "Error during the creation of CL image from buffer");

    return cl::Image2D(cl_image);
}

void handle_cl_error(const std::string &function_name, cl_int error_code)
{
    if (error_code != CL_SUCCESS)
    {
        std::string error_message = function_name + " - Error code: " + std::to_string(error_code);
        ARM_COMPUTE_ERROR(error_message.c_str());
    }
}

} // namespace arm_compute
