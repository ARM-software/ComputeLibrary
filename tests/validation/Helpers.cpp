/*
 * Copyright (c) 2017 ARM Limited.
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
#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TensorShape calculate_depth_concatenate_shape(const std::vector<TensorShape> &input_shapes)
{
    ARM_COMPUTE_ERROR_ON(input_shapes.empty());

    TensorShape out_shape = input_shapes[0];

    size_t max_x = 0;
    size_t max_y = 0;
    size_t depth = 0;

    for(const auto &shape : input_shapes)
    {
        max_x = std::max(shape.x(), max_x);
        max_y = std::max(shape.y(), max_y);
        depth += shape.z();
    }

    out_shape.set(0, max_x);
    out_shape.set(1, max_y);
    out_shape.set(2, depth);

    return out_shape;
}
} // namespace validation
} // namespace test
} // namespace arm_compute
