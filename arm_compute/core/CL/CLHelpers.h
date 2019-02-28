/*
 * Copyright (c) 2016-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLHELPERS_H__
#define __ARM_COMPUTE_CLHELPERS_H__

#include "arm_compute/core/CL/CLTypes.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/Helpers.h"
#include "support/ToolchainSupport.h"

#include <string>

namespace arm_compute
{
enum class DataType;

/** Max vector width of an OpenCL vector */
static constexpr unsigned int max_cl_vector_width = 16;

/** Translates a tensor data type to the appropriate OpenCL type.
 *
 * @param[in] dt @ref DataType to be translated to OpenCL type.
 *
 * @return The string specifying the OpenCL type to be used.
 */
std::string get_cl_type_from_data_type(const DataType &dt);

/** Translates a tensor data type to the appropriate OpenCL select type.
 *
 * @param[in] dt @ref DataType to be translated to OpenCL select type.
 *
 * @return The string specifying the OpenCL select type to be used.
 */
std::string get_cl_select_type_from_data_type(const DataType &dt);

/** Get the size of a data type in number of bits.
 *
 * @param[in] dt @ref DataType.
 *
 * @return Number of bits in the data type specified.
 */
std::string get_data_size_from_data_type(const DataType &dt);

/** Translates fixed point tensor data type to the underlying OpenCL type.
 *
 * @param[in] dt @ref DataType to be translated to OpenCL type.
 *
 * @return The string specifying the underlying OpenCL type to be used.
 */
std::string get_underlying_cl_type_from_data_type(const DataType &dt);

/** Helper function to get the GPU target from CL device
 *
 * @param[in] device A CL device
 *
 * @return the GPU target
 */
GPUTarget get_target_from_device(const cl::Device &device);

/** Helper function to get the highest OpenCL version supported
 *
 * @param[in] device A CL device
 *
 * @return the highest OpenCL version supported
 */
CLVersion get_cl_version(const cl::Device &device);

/** Helper function to check whether a given extension is supported
 *
 * @param[in] device         A CL device
 * @param[in] extension_name Name of the extension to be checked
 *
 * @return True if the extension is supported
 */
bool device_supports_extension(const cl::Device &device, const char *extension_name);

/** Helper function to check whether the cl_khr_fp16 extension is supported
 *
 * @param[in] device A CL device
 *
 * @return True if the extension is supported
 */
bool fp16_supported(const cl::Device &device);
/** Helper function to check whether the arm_non_uniform_work_group_size extension is supported
 *
 * @param[in] device A CL device
 *
 * @return True if the extension is supported
 */
bool arm_non_uniform_workgroup_supported(const cl::Device &device);
/** Helper function to check whether the cl_arm_integer_dot_product_int8 extension is supported
 *
 * @param[in] device A CL device
 *
 * @return True if the extension is supported
 */
bool dot8_supported(const cl::Device &device);

/** Helper function to check whether the cl_arm_integer_dot_product_accumulate_int8 extension is supported
 *
 * @param[in] device A CL device
 *
 * @return True if the extension is supported
 */
bool dot8_acc_supported(const cl::Device &device);

/** This function checks if the Winograd configuration (defined through the output tile, kernel size and the data layout) is supported on OpenCL
 *
 * @param[in] output_tile Output tile for the Winograd filtering algorithm
 * @param[in] kernel_size Kernel size for the Winograd filtering algorithm
 * @param[in] data_layout Data layout of the input tensor
 *
 * @return True if the configuration is supported
 */
bool cl_winograd_convolution_layer_supported(const Size2D &output_tile, const Size2D &kernel_size, DataLayout data_layout);

/** Helper function to get the preferred native vector width size for built-in scalar types that can be put into vectors
 *
 * @param[in] device A CL device
 * @param[in] dt     data type
 *
 * @return preferred vector width
 */
size_t preferred_vector_width(const cl::Device &device, DataType dt);
}
#endif /* __ARM_COMPUTE_CLHELPERS_H__ */
