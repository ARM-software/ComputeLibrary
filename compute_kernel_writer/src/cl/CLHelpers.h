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
#ifndef COMPUTE_KERNEL_WRITER_SRC_CL_CLHELPERS_H
#define COMPUTE_KERNEL_WRITER_SRC_CL_CLHELPERS_H

#include <string>

/** OpenCL specific helper functions */
namespace ckw
{
// Forward declarations
enum class DataType;

/** Helper function to validate the vector length of OpenCL vector data types
 *
 * @param[in] len Vector length
 *
 * @return true if the vector lenght is valid. It returns false, otherwise.
 */
bool cl_validate_vector_length(int32_t len);

/** Helper function to return the OpenCL datatype as a string from a @ref DataType and vector length as int32_t variable
 *
 * @param[in] dt  Datatype
 * @param[in] len Vector length
 *
 * @return the OpenCL datatype as a string
 */
std::string cl_get_variable_datatype_as_string(DataType dt, int32_t len);
} // namespace ckw

#endif /* COMPUTE_KERNEL_WRITER_SRC_CL_CLHELPERS_H */
