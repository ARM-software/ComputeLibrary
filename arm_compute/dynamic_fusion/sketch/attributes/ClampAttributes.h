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

#ifndef ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_ATTRIBUTES_CLAMPATTRIBUTES
#define ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_ATTRIBUTES_CLAMPATTRIBUTES

#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Attributes are backend-agnostic parameters (in addition to the input/output tensors) of an operator.
 */

/** Clamp attributes */
class ClampAttributes
{
public:
    /** Set the minimum clip value */
    ClampAttributes &min_val(const float &min_val);

    /** Get the minimum clip value */
    float min_val() const;

    /** Set the maximum clip value */
    ClampAttributes &max_val(const float &max_val);

    /** Get the maximum clip value */
    float max_val() const;

private:
    float _min_val{};
    float _max_val{};
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_ATTRIBUTES_CLAMPATTRIBUTES */
