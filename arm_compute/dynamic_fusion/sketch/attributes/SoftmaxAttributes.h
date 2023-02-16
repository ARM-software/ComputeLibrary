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
#ifndef ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_ATTRIBUTES_SOFTMAXATTRIBUTES
#define ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_ATTRIBUTES_SOFTMAXATTRIBUTES

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Attributes are backend-agnostic parameters (in addition to the input/output tensors) of an operator.
 */

/** Softmax attributes */
class SoftmaxAttributes
{
public:
    /** Set beta */
    SoftmaxAttributes &beta(float beta);
    /** Get beta */
    float beta() const;
    /** Set is_log */
    SoftmaxAttributes &is_log_softmax(bool is_log);
    /** Get is_log */
    bool is_log_softmax() const;
    /** Set axis */
    SoftmaxAttributes &axis(int axis);
    /** Get axis */
    int axis() const;

private:
    float _beta{ 1.f };      /**< Scaling factor for the exponent */
    bool  _is_log_softmax{}; /**< True if operation is log-softmax */
    int   _axis{};           /**< Axis/Dimension to perform the operation */
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif /* ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_ATTRIBUTES_SOFTMAXATTRIBUTES */
