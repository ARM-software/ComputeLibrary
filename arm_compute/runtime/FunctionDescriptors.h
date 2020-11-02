/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_RUNTIME_FUNCTION_DESCRIPTORS_H
#define ARM_COMPUTE_RUNTIME_FUNCTION_DESCRIPTORS_H

#include "arm_compute/core/Types.h"

#include <utility>

namespace arm_compute
{
/** FFT direction to use */
enum class FFTDirection
{
    Forward,
    Inverse
};

/** Descriptor used by the FFT1D function */
struct FFT1DInfo
{
    unsigned int axis{ 0 };                          /**< Axis to run the FFT on. */
    FFTDirection direction{ FFTDirection::Forward }; /**< Direction of the FFT. */
};

/** Descriptor used by the FFT2D function */
struct FFT2DInfo
{
    unsigned int axis0{ 0 };                         /**< Axis to run first pass on. If same, multiple transforms are performed on single axis*/
    unsigned int axis1{ 1 };                         /**< Axis to run second pass on. If same, multiple transforms are performed on single axis*/
    FFTDirection direction{ FFTDirection::Forward }; /**< Direction of the FFT. */
};

/** Descriptor used by the Convolution function */
struct Conv2dInfo
{
    Conv2dInfo() = default;

    Conv2dInfo(const PadStrideInfo       &conv_info,
               const Size2D              &dilation,
               const ActivationLayerInfo &act_info,
               bool                       enable_fast_math,
               unsigned int               num_groups)
        : conv_info(conv_info), dilation(dilation), act_info(act_info), enable_fast_math(enable_fast_math), num_groups(num_groups)
    {
    }

    PadStrideInfo       conv_info{};
    Size2D              dilation{ 1U, 1U };
    ActivationLayerInfo act_info{};
    bool                enable_fast_math{ false };
    unsigned int        num_groups{ 1 };
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_RUNTIME_FUNCTION_DESCRIPTORS_H */
