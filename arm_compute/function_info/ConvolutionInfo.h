/*
 * Copyright (c) 2016-2023 Arm Limited.
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
#ifndef ACL_ARM_COMPUTE_FUNCTION_INFO_CONVOLUTIONINFO
#define ACL_ARM_COMPUTE_FUNCTION_INFO_CONVOLUTIONINFO

#include "arm_compute/core/CoreTypes.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/function_info/ActivationLayerInfo.h"

namespace arm_compute
{
struct ConvolutionInfo
{
    ConvolutionInfo() = default;
    ConvolutionInfo(const PadStrideInfo       &pad_stride_info,
                    unsigned int               depth_multiplier,
                    const ActivationLayerInfo &act_info,
                    const Size2D              &dilation)
        : pad_stride_info(pad_stride_info), depth_multiplier(depth_multiplier), act_info(act_info), dilation(dilation)
    {
    }
    PadStrideInfo pad_stride_info{}; /**< Convolution info (Pads, strides,...) */
    unsigned int  depth_multiplier{
        1}; /**< Multiplier to apply to input's depth to retrieve the output depth. Defaults to 1 */
    ActivationLayerInfo act_info{};             /**< Fused activation to apply after convolution. */
    Size2D              dilation{Size2D(1, 1)}; /**< Dilation, in elements, across x and y. Defaults to (1, 1). */
};
} // namespace arm_compute
#endif /* ACL_ARM_COMPUTE_FUNCTION_INFO_CONVOLUTIONINFO */
