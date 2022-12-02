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

#ifndef ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_ATTRIBUTES_RESIZEATTRIBUTES
#define ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_ATTRIBUTES_RESIZEATTRIBUTES

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Types.h"
#include <cstdint>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Attributes are backend-agnostic parameters (in addition to the input/output tensors) of an operator.
 */

/** Resize attributes */
class ResizeAttributes
{
public:
    /** Set output width */
    ResizeAttributes &output_width(int32_t output_width);

    /** Get output width */
    int32_t output_width() const;

    /** Set output height */
    ResizeAttributes &output_height(int32_t output_height);

    /** Get output height */
    int32_t output_height() const;

    /** Set interpolation policy */
    ResizeAttributes &interpolation_policy(InterpolationPolicy interpolation_policy);

    /** Get interpolation policy */
    InterpolationPolicy interpolation_policy() const;

    /** Set sampling policy */
    ResizeAttributes &sampling_policy(SamplingPolicy sampling_policy);

    /** Get sampling policy */
    SamplingPolicy sampling_policy() const;

    /** Set align corners */
    ResizeAttributes &align_corners(bool align_corners);

    /** Get align corners */
    bool align_corners() const;

private:
    int32_t             _output_width{};
    int32_t             _output_height{};
    InterpolationPolicy _interpolation_policy{ InterpolationPolicy::BILINEAR };
    SamplingPolicy      _sampling_policy{ SamplingPolicy::CENTER };
    bool                _align_corners{ false };
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif /* ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_ATTRIBUTES_RESIZEATTRIBUTES */
