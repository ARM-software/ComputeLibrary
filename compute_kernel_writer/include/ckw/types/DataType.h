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

#ifndef CKW_INCLUDE_CKW_DATATYPE_H
#define CKW_INCLUDE_CKW_DATATYPE_H

#include <cstdint>

namespace ckw
{

/** Compute Kernel Writer data types. This data type is used by the code variables and tensor arguments. */
enum class DataType : int32_t
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

} // namespace ckw

#endif //CKW_INCLUDE_CKW_DATATYPE_H
