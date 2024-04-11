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
#ifndef SRC_RUNTIME_HEURISTICS_DWC_NATIVE_CLDWCNATIVEHEURISTICSHELPERS
#define SRC_RUNTIME_HEURISTICS_DWC_NATIVE_CLDWCNATIVEHEURISTICSHELPERS

namespace arm_compute
{
// Forward declaration
class ITensorInfo;

namespace cl_dwc
{
/** Utility function to know whether we can use the cl image storage for the weights of depthwise convolution to get better performance
 *
 * @param[in] weights          Weights TensorInfo of the depthwise convolution
 * @param[in] depth_multiplier Depth multiplier
 *
 * @return true if the weights of depthwise convolution can be kept in the cl image storage to improve the performance
 */
bool use_cl_image_for_weights(const ITensorInfo *weights, unsigned int depth_multiplier);

} // namespace cl_dwc
} // namespace arm_compute
#endif /* SRC_RUNTIME_HEURISTICS_DWC_NATIVE_CLDWCNATIVEHEURISTICSHELPERS */
