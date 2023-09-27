/*
 * Copyright (c) 2020-2021, 2023 Arm Limited.
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

#ifndef ACL_SRC_CORE_CL_CLUTILS_H
#define ACL_SRC_CORE_CL_CLUTILS_H

#include "arm_compute/core/CL/OpenCL.h"

#include <map>

namespace arm_compute
{
class TensorShape;
class CLBuildOptions;
class ITensorInfo;
class ICLTensor;
enum class DataType;

/** OpenCL Image2D types */
enum class CLImage2DType
{
    ReadOnly,
    WriteOnly
};

/** Create a cl::Image2D object from a tensor
 *
 * @param[in] tensor     Tensor from which to construct Image 2D object
 * @param[in] image_type Image 2D type (@ref CLImage2DType)
 *
 * @return cl::Image2D object
 */
cl::Image2D create_image2d_from_tensor(const ICLTensor *tensor, CLImage2DType image_type);

/** Create a cl::Image2D object from an OpenCL buffer
 *
 * @note The following conditions are required to create a OpenCL image object from OpenCL buffer,
 *       -# The platform should support the OpenCL cl_khr_image2d_from_buffer extension
 *       -# The stride Y for the input1 should satisfy the OpenCL pitch alignment requirement
 *       -# input width should be less or equal to (CL_DEVICE_IMAGE2D_MAX_WIDTH * 4)
 *       -# input height should be less or equal to CL_DEVICE_IMAGE2D_MAX_HEIGHT
 *
 * It is user responsibility to ensure the above conditions are satisfied since no checks are performed within this function
 *
 * @param[in] ctx             cl::Context object
 * @param[in] buffer          cl::Buffer object from which the OpenCL image2d object is created
 * @param[in] shape2d         2D tensor shape
 * @param[in] data_type       DataType to use. Only supported: F32,F16
 * @param[in] image_row_pitch Image row pitch (a.k.a. stride Y) to be used in the image2d object
 * @param[in] image_type      Image 2D type (@ref CLImage2DType)
 *
 * @return cl::Image2D object
 */
cl::Image2D create_image2d_from_buffer(const cl::Context &ctx,
                                       const cl::Buffer  &buffer,
                                       const TensorShape &shape2d,
                                       DataType           data_type,
                                       size_t             image_row_pitch,
                                       CLImage2DType      image_type);

/** Check for CL error code and throw exception accordingly.
 *
 * @param[in] function_name The name of the CL function being called.
 * @param[in] error_code    The error returned by the CL function.
 */
void handle_cl_error(const std::string &function_name, cl_int error_code);

} // namespace arm_compute

#endif // ACL_SRC_CORE_CL_CLUTILS_H
