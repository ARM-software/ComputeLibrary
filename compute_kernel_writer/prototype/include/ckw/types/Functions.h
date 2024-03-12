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

#ifndef CKW_INCLUDE_CKW_FUNCTIONS_H
#define CKW_INCLUDE_CKW_FUNCTIONS_H

#include <cstdint>

namespace ckw
{

enum class UnaryFunction : int32_t
{
    Exp   = 0x0000,
    Tanh  = 0x0001,
    Sqrt  = 0x0002,
    Erf   = 0x0003,
    Fabs  = 0x0004,
    Log   = 0x0006,
    Round = 0x0007,
    Floor = 0x0008,

    // Misc
    SizeOf = 0x0009,
};

enum class BinaryFunction : int32_t
{
    Min = 0x0000,
    Max = 0x0001,
};

enum class TernaryFunction : int32_t
{
    Select = 0x0000,
    Clamp  = 0x0001,
};

} // namespace ckw

#endif //CKW_INCLUDE_CKW_FUNCTIONS_H
