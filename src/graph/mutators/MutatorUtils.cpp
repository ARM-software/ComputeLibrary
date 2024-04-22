/*
 * Copyright (c) 2021 Arm Limited.
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
#include "src/graph/mutators/MutatorUtils.h"

namespace arm_compute
{
namespace graph
{
bool is_padding_in_height_or_width(const DataLayout &layout, const PaddingList &padding_list)
{
    if (layout == DataLayout::NCHW || layout == DataLayout::NHWC)
    {
        const unsigned int height_index = get_dimension_idx(layout, DataLayoutDimension::HEIGHT);
        const unsigned int width_index  = get_dimension_idx(layout, DataLayoutDimension::WIDTH);

        for (unsigned int i = 0; i < padding_list.size(); ++i)
        {
            if (i != height_index && i != width_index && padding_list[i] != PaddingInfo(0, 0))
            {
                // if the index is not either height or width, don't fuse
                return false;
            }
        }

        return true;
    }

    return false;
}
} // namespace graph
} // namespace arm_compute
