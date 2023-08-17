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

#ifndef CKW_INCLUDE_CKW_TYPES_TENSORSAMPLERTYPES_H
#define CKW_INCLUDE_CKW_TYPES_TENSORSAMPLERTYPES_H

#include <cstdint>

namespace ckw
{

// This enum class defines how the dimensions of a 3d tensor is mapped into x,y and z coordianates.
enum class TensorSamplerFormat : int32_t
{
    Unknown          = 0,
    Dim0_Dim1xDim2_1 = 1, // Original dimensions 1 and 2 are collapsed onto y-axis
    Dim0_Dim1_Dim2   = 2  // Original dimensions stays as they're defined. No collapsing.
};

/** Tensor sampler address mode enum class for X dimension
 *
 *  The following address modes are available in total:
 *      Unknown
 *      None                 : The user guarantees that the coordinate is always in-bound
 *      OverlappingMin       : (FIXED shapes only) Reduce the load/store length when x == 0 (MIN). The load length will be width % original length
 *                             Leftover elements can be handled using overlapping. This involves processing some of the elements in the array twice.
 *      ClampToBorderMaxOnly : Clamp to max value allowed in the corresponding dimension, and construct an if/else guard to prevent out of bound access,
 *                             e.g. if( y < size-of-dimension-y ){ <do the operation>  }
 *      SkipLessThanZero     : Skip loading/storing if the index is less than 0
 *
 *  Individual dimensions choose which adddress mode to implement in their respective enum classes.
 */
enum class TensorSamplerAddressModeX : int32_t
{
    Unknown        = 0,
    None           = 1,
    OverlappingMin = 2
};

/**
 * Similar to @ref TensorSamplerAddressModeX
 */
enum class TensorSamplerAddressModeY : int32_t
{
    Unknown              = 0,
    None                 = 1,
    OverlappingMin       = 2,
    ClampToBorderMaxOnly = 3,
    SkipLessThanZero     = 4
};

/**
 * Similar to @ref TensorSamplerAddressModeX
 */
enum class TensorSamplerAddressModeZ : int32_t
{
    Unknown        = 0,
    None           = 1,
};

} // namespace ckw

#endif // CKW_INCLUDE_CKW_TYPES_TENSORSAMPLERTYPES_H
