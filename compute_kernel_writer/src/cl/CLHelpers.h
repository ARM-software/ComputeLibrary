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
#ifndef CKW_SRC_CL_CLHELPERS_H
#define CKW_SRC_CL_CLHELPERS_H

#include "ckw/types/Operators.h"

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

/** OpenCL specific helper functions */
namespace ckw
{
// Forward declarations
enum class DataType;
enum class TensorStorageType : uint32_t;

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

/** Return the assignment operator in OpenCL language.
 *
 * @param[in] op The assignment operator.
 *
 * @return The operator in OpenCL language as a string.
 */
std::string cl_get_assignment_op_as_string(AssignmentOp op);

/** Return the information about the unary operation.
 *
 * The result contains:
 *   - is_func: true if it's a function and false if it's an unary operator in OpenCL language.
 *   - str: the function name or the operator in OpenCL language.
 *
 * @param[in] op The unary operator.
 *
 * @return The information about the unary operation.
 */
std::tuple<bool, std::string> cl_get_unary_op(UnaryOp op);

/** Return the information about the binary operation.
 *
 * The result contains:
 *   - is_func: true if it's a function and false if it's an binary operator in OpenCL language.
 *   - str: the function name or the operator in OpenCL language.
 *
 * @param[in] op        The binary operator.
 * @param[in] data_type The input data type.
 *
 * @return The information about the binary operation.
 */
std::tuple<bool, std::string> cl_get_binary_op(BinaryOp op, DataType data_type);

/** Return the information about the ternary operation.
 *
 * The result contains:
 *   - is_func: true if it's a function and false if it's a ternary operator in OpenCL language.
 *   - str: the function name or the operator in OpenCL language.
 *
 * @param[in] op The ternary operator.
 *
 * @return The information about the ternary operation.
 */
std::tuple<bool, std::string> cl_get_ternary_op(TernaryOp op);

/** Helper function to return the OpenCL vector size that accommodate the the desired width
 *
 * @param[in] width The desired width
 *
 * @return the OpenCL vector size
*/
int32_t cl_round_up_to_nearest_valid_vector_width(int32_t width);

/** Helper function to return the OpenCL storage type as a string from a @ref TensorStorage
 *
 * @param[in] storage Storage type
 *
 * @return the OpenCL storage type as a string
 */
std::string cl_get_variable_storagetype_as_string(TensorStorageType storage);

/** Helper function to decompose a vector width into a summation of valid OpenCL vector widths.
 *
 * @param[in] vector_width Vector width to be decomposed
 *
 * @return a vector of OpenCL vector widths
 */
std::vector<int32_t> cl_decompose_vector_width(int32_t vector_width);

/** Helper function to get OpenCL data type from the data type enum and width
 *  It'll round up the given vector width to the nearest valid OpenCL vector width.
 *
 *  @param[in] dt    data type enum
 *  @param[in] width vector width
 *
 * @return a string representation of the data type
 */
std::string cl_data_type_rounded_up_to_valid_vector_width(DataType dt, int32_t width);
} // namespace ckw

#endif /* CKW_SRC_CL_CLHELPERS_H */
