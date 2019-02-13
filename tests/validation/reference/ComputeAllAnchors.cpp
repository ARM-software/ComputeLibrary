/*
 * Copyright (c) 2019 ARM Limited.
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
#include "ComputeAllAnchors.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/misc/Utility.h"
#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> compute_all_anchors(const SimpleTensor<T> &anchors, const ComputeAnchorsInfo &info)
{
    const int   num_anchors = anchors.shape()[1];
    const auto  width       = int(info.feat_width());
    const auto  height      = int(info.feat_height());
    const float stride      = 1. / info.spatial_scale();

    SimpleTensor<T> all_anchors(TensorShape(4, width * height * num_anchors), anchors.data_type());
    const T        *anchors_ptr     = anchors.data();
    T              *all_anchors_ptr = all_anchors.data();

    // Iterate over the input grid and anchors
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            for(int a = 0; a < num_anchors; a++)
            {
                const T      shift_x   = T(x) * T(stride);
                const T      shift_y   = T(y) * T(stride);
                const size_t anchor_id = a + x * num_anchors + y * width * num_anchors;
                // x1
                all_anchors_ptr[anchor_id * 4] = anchors_ptr[4 * a] + shift_x;
                // y1
                all_anchors_ptr[anchor_id * 4 + 1] = anchors_ptr[4 * a + 1] + shift_y;
                // x2
                all_anchors_ptr[anchor_id * 4 + 2] = anchors_ptr[4 * a + 2] + shift_x;
                // y2
                all_anchors_ptr[anchor_id * 4 + 3] = anchors_ptr[4 * a + 3] + shift_y;
            }
        }
    }
    return all_anchors;
}
template SimpleTensor<float> compute_all_anchors(const SimpleTensor<float> &anchors, const ComputeAnchorsInfo &info);
template SimpleTensor<half> compute_all_anchors(const SimpleTensor<half> &anchors, const ComputeAnchorsInfo &info);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
