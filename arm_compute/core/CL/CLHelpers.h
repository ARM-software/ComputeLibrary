/*
 * Copyright (c) 2016, 2017 ARM Limited.
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

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "support/ToolchainSupport.h"

#include <string>

namespace arm_compute
{
enum class DataType;
enum class GPUTarget;

/** Enable operation operations on GPUTarget enumerations */
template <>
struct enable_bitwise_ops<arm_compute::GPUTarget>
{
    static constexpr bool value = true;
};

/** Max vector width of an OpenCL vector */
static constexpr unsigned int max_cl_vector_width = 16;

/** Translates a tensor data type to the appropriate OpenCL type.
 *
 * @param[in] dt @ref DataType to be translated to OpenCL type.
 *
 * @return The string specifying the OpenCL type to be used.
 */
std::string get_cl_type_from_data_type(const DataType &dt);

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

/** Translates a given gpu device target to string.
 *
 * @param[in] target Given gpu target.
 *
 * @return The string describing the target.
 */
const std::string &string_from_target(GPUTarget target);

/** Helper function to create and return a unique_ptr pointed to a CL kernel object
 *  It also calls the kernel's configuration.
 *
 * @param[in] args All the arguments that need pass to kernel's configuration.
 *
 * @return A unique pointer pointed to a CL kernel object
 */
template <typename Kernel, typename... T>
std::unique_ptr<Kernel> create_configure_kernel(T &&... args)
{
    std::unique_ptr<Kernel> k = arm_compute::support::cpp14::make_unique<Kernel>();
    k->configure(std::forward<T>(args)...);
    return k;
}

/** Helper function to create and return a unique_ptr pointed to a CL kernel object
 *
 * @return A unique pointer pointed to a CL kernel object
 */
template <typename Kernel>
std::unique_ptr<Kernel> create_kernel()
{
    std::unique_ptr<Kernel> k = arm_compute::support::cpp14::make_unique<Kernel>();
    return k;
}

/** Helper function to get the GPU target from CL device
 *
 * @param[in] device A CL device
 *
 * @return the GPU target
 */
GPUTarget get_target_from_device(cl::Device &device);

/** Helper function to get the GPU arch
 *
 * @param[in] target GPU target
 *
 * @return the GPU target which shows the arch
 */
GPUTarget get_arch_from_target(GPUTarget target);

/** Helper function to get the highest OpenCL version supported
 *
 * @param[in] device A CL device
 *
 * @return the highest OpenCL version supported
 */
CLVersion get_cl_version(const cl::Device &device);
/** Helper function to check whether the cl_khr_fp16 extension is supported
 *
 * @param[in] device A CL device
 *
 * @return True if the extension is supported
 */
bool fp16_support(const cl::Device &device);
/** Helper function to check whether the arm_non_uniform_work_group_size extension is supported
 *
 * @param[in] device A CL device
 *
 * @return True if the extension is supported
 */
bool non_uniform_workgroup_support(const cl::Device &device);
}
#endif /* __ARM_COMPUTE_CLHELPERS_H__ */
