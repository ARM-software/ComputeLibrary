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
#include "activation_float_helpers.h"
#include "helpers.h"
#include "tile_helpers.h"

#ifdef BIAS
// This function performs in-place bias addition for integer datatype when bias is enabled.
// Note The tile's dimensions used for the LHS and RHS matrices (M0, N0) must be passed at compile time using -DN0, -DM0 (e.g. -DN0=8, -DM0=4).
inline void perform_bias_addition(uchar *bias_ptr, uint bias_offset_first_element_in_bytes, TILE(int, M0, N0, acc), uint x)
{
    TILE(int, 1, N0, bias_tile);

    // below expands to use bias_ptr and bias_offset_first_element_in_bytes
    T_LOAD(int, 1, N0, BUFFER, bias, x, 0, 1, 0, bias_tile);

    // c = c + bias[broadcasted]
    T_ELTWISE_BROADCAST_ADD_X(int, M0, N0, acc, bias_tile, acc);
}
#endif // defined(BIAS)

#if defined(MAT_MUL_NATIVE_QUANTIZED_MMUL_NT_NT)
/** This OpenCL kernel performs the batch matrix multiplication (BatchMatMul): LHS non-transposed, RHS non-transposed - buffer only
 *
 * TODO: report build configuration
 *
 * @param[in]  lhs_ptr                            Pointer to the lhs matrix. Supported data types: QASYMM8_SIGNED/QASYMM8
 * @param[in]  lhs_stride_y                       Stride of the lhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  lhs_stride_z                       Stride of the lhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  lhs_w                              The width of the lhs tensor
 * @param[in]  lhs_h                              The height of the lhs tensor
 * @param[in]  lhs_n                              Number of the matrices (buffers) in the batch
 * @param[in]  lhs_offset_first_element_in_bytes  The offset of the first element in the lhs matrix
 * @param[in]  rhs_ptr                            Pointer to the rhs matrix. Supported data types: same as @p lhs_ptr
 * @param[in]  rhs_stride_y                       Stride of the rhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  rhs_stride_z                       Stride of the rhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  rhs_w                              The width of the rhs tensor
 * @param[in]  rhs_h                              The height of the rhs tensor
 * @param[in]  rhs_n                              Number of the matrices (buffers) in the batch
 * @param[in]  rhs_offset_first_element_in_bytes  The offset of the first element in the rhs matrix
 * @param[in]  bias_ptr                           (Optional) Pointer to the bias tensor. Supported data type: same as @p lhs_ptr
 * @param[in]  bias_stride_y                      (Optional) Stride of the bias tensor in Y dimension (in bytes)
 * @param[in]  bias_stride_z                      (Optional) Stride of the bias tensor in Z dimension (in bytes)
 * @param[in]  bias_w                             (Optional) The size of the width dimension of the bias tensor
 * @param[in]  bias_h                             (Optional) The size of the height dimension of the bias tensor
 * @param[in]  bias_n                             (Optional) The size of the depth dimension of the bias tensor
 * @param[in]  bias_offset_first_element_in_bytes (Optional) The offset of the first element in the bias tensor
 * @param[out] dst_ptr                            Pointer to the dst matrix. Supported data types: same as @p lhs_ptr
 * @param[in]  dst_stride_y                       Stride of the dst matrix in Y (2nd) dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the dst tensor in Z (3rd) dimension (in bytes)
 * @param[in]  dst_w                              The width of the dst tensor
 * @param[in]  dst_h                              The height of the dst tensor
 * @param[in]  dst_n                              Number of the matrices (buffers) in the batch
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the dst matrix
 */
__kernel void mat_mul_native_quantized_mmul_nt_nt(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, BUFFER),
#ifdef BIAS
    TENSOR3D_T(bias, BUFFER),
#endif // defined(BIAS)
    TENSOR3D_T(dst, BUFFER))
{
}
#endif // defined(MAT_MUL_NATIVE_QUANTIZED_MMUL_NT_NT)

#if defined(MAT_MUL_NATIVE_QUANTIZED_MMUL_NT_T)
/** This OpenCL kernel performs the batch matrix multiplication (BatchMatMul): LHS non-transposed, RHS transposed - buffer only
 *
 * Supported block configurations:
 *     TODO: Report supported M0, N0, K0
 *
 * Similar to mat_mul_native_quantized_mmul_nt_nt()
 */
__kernel void mat_mul_native_quantized_mmul_nt_t(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, BUFFER),
#ifdef BIAS
    TENSOR3D_T(bias, BUFFER),
#endif // defined(BIAS)
    TENSOR3D_T(dst, BUFFER))
{
}
#endif // defined(MAT_MUL_NATIVE_QUANTIZED_MMUL_NT_T)

#if defined(MAT_MUL_NATIVE_QUANTIZED_MMUL_T_NT)
/** This OpenCL kernel performs the batch matrix multiplication (BatchMatMul): LHS transposed, RHS non-transposed
 *
 * Supported block configurations:
 *     TODO: Report supported M0, N0, K0
 *
 * Similar to mat_mul_native_quantized_mmul_nt_nt()
 */
__kernel void mat_mul_native_quantized_mmul_t_nt(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, BUFFER),
#ifdef BIAS
    TENSOR3D_T(bias, BUFFER),
#endif // defined(BIAS)
    TENSOR3D_T(dst, BUFFER))
{
}
#endif // defined(MAT_MUL_NATIVE_QUANTIZED_MMUL_T_NT)

#if defined(MAT_MUL_NATIVE_QUANTIZED_MMUL_T_T)
/** This OpenCL kernel performs the batch matrix multiplication (BatchMatMul): LHS transposed, RHS transposed
 *
 * Supported block configurations:
 *     TODO: Report supported M0, N0, K0
 *
 * Similar to mat_mul_native_quantized_mmul_nt_nt()
 */
__kernel void mat_mul_native_quantized_mmul_t_t(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, BUFFER),
#ifdef BIAS
    TENSOR3D_T(bias, BUFFER),
#endif // defined(BIAS)
    TENSOR3D_T(dst, BUFFER))
{
}
#endif // defined(MAT_MUL_NATIVE_QUANTIZED_MMUL_T_T)
