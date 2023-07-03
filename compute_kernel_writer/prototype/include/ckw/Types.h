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

#ifndef CKW_PROTOTYPE_INCLUDE_CKW_TYPES_H
#define CKW_PROTOTYPE_INCLUDE_CKW_TYPES_H

#include <array>
#include <cstdint>

namespace ckw
{

/** Compute Kernel Writer data types. This data type is used by the code variables and tensor arguments. */
enum class DataType
{
    Unknown = 0x00,
    Fp32    = 0x11,
    Fp16    = 0x12,
    Int32   = 0x21,
    Int16   = 0x22,
    Int8    = 0x24,
    Uint32  = 0x31,
    Uint16  = 0x32,
    Uint8   = 0x34,
    Bool    = 0x41
};

enum class GpuTargetLanguage
{
    Unknown,
    OpenCL
};

/* Binary operations
*/
enum class BinaryOp : int32_t
{
    // Elementwise
    Add = 0x0000, // +
    Sub = 0x0001, // -
    Mul = 0x0002, // *
    Div = 0x0003, // /
    Mod = 0x0004, // %
    // Relational
    Equal        = 0x1000, // ==
    Less         = 0x1001, // <
    LessEqual    = 0x1002, // <=
    Greater      = 0x1003, // >
    GreaterEqual = 0x1004, // >=
    // Algebra
    MatMul_Nt_Nt = 0x2000, // X
    MatMul_Nt_T  = 0x2001, // X
    MatMul_T_Nt  = 0x2002, // X
    MatMul_T_T   = 0x2003, // X
    Dot          = 0x2004, // .
    // Logical
    LogicalAnd = 0x3000, // &&
    LogicalOr  = 0x3001, // ||
    LogicalNot = 0x3002  // !
};

enum class AssignmentOp : int32_t
{
    // Unary
    Increment = 0x0000, // +=
    Decrement = 0x0001, // -=
};

enum class ScalarUnaryFunction : int32_t
{
    Exp,
};

enum class TensorSamplerFormat : int32_t
{
    Unknown = 0,
    C_WH_1  = 1,
    C_W_H   = 2
};

enum class TensorSamplerAddressModeX : int32_t
{
    Unknown        = 0,
    None           = 1, // The user guarantees that the X coordinate is always in-bound
    OverlappingMin = 2  // (FIXED shapes only) Reduce the load/store length when x == 0 (MIN). The load length will be width % original length
                        // Leftover elements can be handled using overlapping. This involves processing some of the elements in the array twice.
};

enum class TensorSamplerAddressModeY : int32_t
{
    Unknown                  = 0,
    None                     = 1, // The user guarantees that the Y coordinate is always in-bound
    OverlappingMin           = 2, // (FIXED shapes only) Reduce the load/store length when x == 0 (MIN). The load length will be width % original length
    Skip                     = 3, // Skip the read/write
    SkipMinEdgeOnly          = 4, // Skip greater than or equal to max only. The user guarantees that the Y coordinate is always >= 0
    SkipMaxEdgeOnly          = 5, // Skip less than 0 only
    ClampToNearest           = 6, // Clamp the coordinate to nearest edge (0 or max value allowed on Y)
    ClampToMinEdgeOnly       = 7, // Clamp the negative coordinate to 0 only. Therefore, we expect Y to be always < MAX
    ClampToMaxEdgeOnly       = 8, // Clamp the coordinate to the max value allowed on Y only. We expect Y to be always >= 0
    ClampToBorder            = 9, // Clamp to border which always has 0 value
    ClampToBorderMinEdgeOnly = 10,
    ClampToBorderMaxEdgeOnly = 11
};

enum class TensorSamplerAddressModeZ : int32_t
{
    Unknown            = 0,
    None               = 1, // The user guarantees that the Y coordinate is always in-bound
    Skip               = 3, // Skip the read/write
    SkipMinEdgeOnly    = 4, // Skip greater than or equal to max only. The user guarantees that the Y coordinate is always >= 0
    SkipMaxEdgeOnly    = 5, // Skip less than 0 only
    ClampToNearest     = 6, // Clamp the coordinate to nearest edge (0 or max value allowed on Y)
    ClampToMinEdgeOnly = 7, // Clamp the negative coordinate to 0 only. Therefore, we expect Y to be always < MAX
    ClampToMaxEdgeOnly = 8, // Clamp the coordinate to the max value allowed on Y only. We expect Y to be always >= 0
};

} // namespace ckw

#endif // CKW_PROTOTYPE_INCLUDE_CKW_TYPES_H
