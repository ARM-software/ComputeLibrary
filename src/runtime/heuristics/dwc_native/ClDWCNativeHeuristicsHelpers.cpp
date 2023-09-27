/*
 * Copyright (c) 2022 Arm Limited.
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
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"

namespace arm_compute
{
namespace cl_dwc
{
bool use_cl_image_for_weights(const ITensorInfo *weights, unsigned int depth_multiplier)
{
    // Check whether we can use the cl image with the weights.
    if (!export_to_cl_image(weights))
    {
        return false;
    }

    const size_t idx_w    = get_data_layout_dimension_index(weights->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_h    = get_data_layout_dimension_index(weights->data_layout(), DataLayoutDimension::HEIGHT);
    const size_t kernel_w = weights->tensor_shape()[idx_w];
    const size_t kernel_h = weights->tensor_shape()[idx_h];

    // If we can use the cl image storage with the weights, we prefer to use the cl buffer storage in the following cases for performance reasons:
    // 1- When the kernel size is 1x1
    // 2- When the depth multiplier is greater than 1 and not multiple of 4.
    if ((kernel_w == 1) && (kernel_h == 1))
    {
        return false;
    }

    if ((depth_multiplier > 1) && (depth_multiplier % 4) != 0)
    {
        return false;
    }

    return true;
}
} // namespace cl_dwc
} // namespace arm_compute
