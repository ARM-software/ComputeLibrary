/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_TYPE_PRINTER_H
#define ARM_COMPUTE_TEST_TYPE_PRINTER_H

#include "tests/Types.h"

namespace arm_compute
{
/** Formatted output of the GradientDimension type.
 *
 * @param[out] os  Output stream
 * @param[in]  dim Type to output
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const GradientDimension &dim)
{
    switch(dim)
    {
        case GradientDimension::GRAD_X:
            os << "GRAD_X";
            break;
        case GradientDimension::GRAD_Y:
            os << "GRAD_Y";
            break;
        case GradientDimension::GRAD_XY:
            os << "GRAD_XY";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

/** Formatted output of the GradientDimension type.
 *
 * @param[in] type Type to output
 *
 * @return Formatted string.
 */
inline std::string to_string(const arm_compute::GradientDimension &type)
{
    std::stringstream str;
    str << type;
    return str.str();
}

} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_TYPE_PRINTER_H */
