/*
 * Copyright (c) 2016-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CLHELPERS_H
#define ARM_COMPUTE_CLHELPERS_H

#include "arm_compute/core/CL/CLTypes.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Types.h"

#include <set>
#include <string>

namespace arm_compute
{
class CLCompileContext;
class CLBuildOptions;

enum class DataType;

/** Max vector width of an OpenCL vector */
static constexpr unsigned int max_cl_vector_width = 16;

/** Max number of manual loop unrolling */
static constexpr int max_manual_loop_unrolling = 128;

/** Translates a tensor data type to the appropriate OpenCL type.
 *
 * @param[in] dt @ref DataType to be translated to OpenCL type.
 *
 * @return The string specifying the OpenCL type to be used.
 */
std::string get_cl_type_from_data_type(const DataType &dt);

/** Translates a tensor data type to the appropriate OpenCL promoted type.
 *
 * @param[in] dt @ref DataType to be used to get the promoted OpenCL type.
 *
 * @return The string specifying the OpenCL type to be used.
 */
std::string get_cl_promoted_type_from_data_type(const DataType &dt);

/** Translates the element size to an unsigned integer data type
 *
 * @param[in] element_size Size in bytes of an element.
 *
 * @return The string specifying the OpenCL type to be used.
 */
std::string get_cl_unsigned_type_from_element_size(size_t element_size);

/** Translates the element size to an signed integer data type
 *
 * @param[in] element_size Size in bytes of an element.
 *
 * @return The string specifying the OpenCL type to be used.
 */
std::string get_cl_signed_type_from_element_size(size_t element_size);

/** Translates a tensor data type to the appropriate OpenCL select type.
 *
 * @param[in] dt @ref DataType to be translated to OpenCL select type.
 *
 * @return The string specifying the OpenCL select type to be used.
 */
std::string get_cl_select_type_from_data_type(const DataType &dt);

/** Translates a tensor data type to the appropriate OpenCL dot8 accumulator type.
 *
 * @param[in] dt @ref DataType to be translated to OpenCL dot8 accumulator type.
 *
 * @return The string specifying the OpenCL dot8 accumulator type to be used.
 */
std::string get_cl_dot8_acc_type_from_data_type(const DataType &dt);

/** Get the size of a data type in number of bits.
 *
 * @param[in] dt @ref DataType.
 *
 * @return Number of bits in the data type specified.
 */
std::string get_data_size_from_data_type(const DataType &dt);

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

/** Helper function to get the cl_image pitch alignment in pixels
 *
 * @param[in] device A CL device
 *
 * @return the cl_image pitch alignment in pixels. If an error occurs, the function will return 0
 */
size_t get_cl_image_pitch_alignment(const cl::Device &device);

/** Helper function to check whether non-uniform work group is supported
 *
 * @param[in] device A CL device
 *
 * @return True if the feature is supported
 */
bool get_cl_non_uniform_work_group_supported(const cl::Device &device);

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

/** Helper function to check if "dummy work-items" are preferred to have a power of two NDRange
 * In case dummy work-items is enabled, it is OpenCL kernel responsibility to check if the work-item is out-of range or not
 *
 * @param[in] device A CL device
 *
 * @return True if dummy work-items should be preferred to dispatch the NDRange
 */
bool preferred_dummy_work_items_support(const cl::Device &device);

/** Helper function to check whether the cl_khr_image2d_from_buffer extension is supported
 *
 * @param[in] device A CL device
 *
 * @return True if the extension is supported
 */
bool image2d_from_buffer_supported(const cl::Device &device);

/** Creates an opencl kernel using a compile context
 *
 * @param[in] ctx         A compile context to be used to create the opencl kernel.
 * @param[in] kernel_name The kernel name.
 * @param[in] build_opts  The build options to be used for the opencl kernel compilation.
 *
 * @return An opencl kernel
 */
cl::Kernel create_kernel(const CLCompileContext &ctx, const std::string &kernel_name, const std::set<std::string> &build_opts = std::set<std::string>());

/** Creates a suitable LWS hint object for parallel implementations. Sets the number of WG based on the input size.
 *  If input width is smaller than 128 we can use fewer threads than 8.
 *
 * @param[in] input_dimension number of elements along the dimension to apply the parallellization
 * @param[in] vector_size     size of the vector in OpenCL
 *
 * @return An LWS hint object
 */
cl::NDRange create_lws_hint_parallel_implementations(unsigned int input_dimension, unsigned int vector_size);

/* Helper function to check if the workgroup batch size modifier parameter is supported on the cl device
 *
 * @param[in] device cl device to check for support
 *
 * @return true if the workgroup batch size modifier parameter is supported, false otherwise
 */
bool get_wbsm_support_info(const cl::Device &device);

/* Helper function to set the workgroup batch size modifier parameter in the kernel
 *
 * @param[in] kernel    cl kernel to set the workgroup batch size modifier parameter
 * @param[in] wbsm_hint workgroup batch size modifier to use
 */
void set_wbsm(cl::Kernel &kernel, cl_int wbsm_hint);

/* Helper function to check if we can export the tensor to cl_image
 *
 * @param[in] input tensor
 *
 * @return true if we can export the tensor to cl_image
 */
bool export_to_cl_image(const ITensorInfo *tensor);

/* Helper function to force unroll with pragma when any of the input values (iterations) are greater than @ref max_manual_loop_unrolling
 *
 * This function passes UNROLL_WITH_PRAGMA at compile time when any of the input values are greater than @ref max_manual_loop_unrolling
 *
 * @param[in] built_opts OpenCL kernel build options
 * @param[in] values     Input values (iterations)
 *
 */
void set_unroll_with_pragma(CLBuildOptions &built_opts, std::initializer_list<int> values);

/** Helper function to check whether the cl_arm_matrix_multiply extension is supported
 *
 * @param[in] device A CL device
 *
 * @return True if the extension is supported
 */
bool arm_matrix_multiply_supported(const cl::Device &device);
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLHELPERS_H */
