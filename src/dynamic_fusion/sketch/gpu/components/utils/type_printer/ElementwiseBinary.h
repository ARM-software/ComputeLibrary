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
#ifndef ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_UTILS_TYPE_PRINTER_ELEMENTWISEBINARY
#define ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_UTILS_TYPE_PRINTER_ELEMENTWISEBINARY

#include "src/dynamic_fusion/sketch/gpu/components/cl/ClComponentElementwiseBinary.h"

#include <ostream>
#include <sstream>
#include <string>

namespace arm_compute
{
/** Type printers for all types related to the component @ref ClComponentElementwiseBinary
 */

using namespace experimental::dynamic_fusion;

/** Formatted output of the pute::experimental::dynamic_fusion::ClComponentElementwiseBinary::Attributes::ElementwiseOp type.
 *
 * @param[out] os Output stream.
 * @param[in]  op arm_compute::experimental::dynamic_fusion::ClComponentElementwiseBinary::Attributes::ElementwiseOp type to output.
 *
 * @return Modified output stream.
 */
inline ::std::ostream &operator<<(::std::ostream &os, const ClComponentElementwiseBinary::Attributes::ElementwiseOp &op)
{
    const std::map<ClComponentElementwiseBinary::Attributes::ElementwiseOp, std::string> op_name =
    {
        { ClComponentElementwiseBinary::Attributes::ElementwiseOp::Add, "add" },
        { ClComponentElementwiseBinary::Attributes::ElementwiseOp::Div, "div" },
        { ClComponentElementwiseBinary::Attributes::ElementwiseOp::Max, "max" },
        { ClComponentElementwiseBinary::Attributes::ElementwiseOp::Min, "min" },
        { ClComponentElementwiseBinary::Attributes::ElementwiseOp::Mul, "mul" },
        { ClComponentElementwiseBinary::Attributes::ElementwiseOp::Power, "power" },
        { ClComponentElementwiseBinary::Attributes::ElementwiseOp::Prelu, "prelu" },
        { ClComponentElementwiseBinary::Attributes::ElementwiseOp::SquaredDiff, "squareddiff" },
        { ClComponentElementwiseBinary::Attributes::ElementwiseOp::Sub, "sub" }
    };
    os << op_name.at(op);
    return os;
}
/** Formatted output of the arm_compute::experimental::dynamic_fusion::ClComponentElementwiseBinary::Attributes::ElementwiseOp type.
 *
 * @param[in] op arm_compute::experimental::dynamic_fusion::ClComponentElementwiseBinary::Attributes::ElementwiseOp type to output.
 *
 * @return Formatted string.
 */
inline std::string to_string(const ClComponentElementwiseBinary::Attributes::ElementwiseOp &op)
{
    std::stringstream str;
    str << op;
    return str.str();
}
} // namespace arm_compute
#endif /* ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_COMPONENTS_UTILS_TYPE_PRINTER_ELEMENTWISEBINARY */
