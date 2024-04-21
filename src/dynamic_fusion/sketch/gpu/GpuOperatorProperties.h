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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUOPERATORPROPERTIES
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUOPERATORPROPERTIES

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Contain properties common to all operator types */

/** Operator type in the context of fusion
 */
enum class GpuOperatorType
{
    /** Simple operators are operators that:
     *  1. Have a 1-to-1 mapping between the input elements and output elements, like elementwise
     *  2. Have exactly 1 output
     */
    Simple,
    /** Complex operators are operators that are not simple but are still fusable with simple ones
     */
    Complex,
    /** Unfusable operators are operators that cannot be fused with any other types of operators
     */
    Unfusable
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUOPERATORPROPERTIES */
