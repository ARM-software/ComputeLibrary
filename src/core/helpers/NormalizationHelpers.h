/*
* Copyright (c) 2020 Arm Limited.
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
#ifndef SRC_CORE_HELPERS_NORMALIZATIONHELPERS_H
#define SRC_CORE_HELPERS_NORMALIZATIONHELPERS_H

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
/** Calculate the normalization dimension index for a given normalization type
 *
 * @param[in] layout Data layout of the input and output tensor
 * @param[in] info   Normalization info
 *
 * @return Normalization dimension index
 */
inline unsigned int get_normalization_dimension_index(DataLayout layout, const NormalizationLayerInfo &info)
{
    const unsigned int width_idx   = get_data_layout_dimension_index(layout, DataLayoutDimension::WIDTH);
    const unsigned int channel_idx = get_data_layout_dimension_index(layout, DataLayoutDimension::CHANNEL);

    return info.is_in_map() ? width_idx : channel_idx;
}
} // namespace arm_compute
#endif /* SRC_CORE_HELPERS_NORMALIZATIONHELPERS_H */
