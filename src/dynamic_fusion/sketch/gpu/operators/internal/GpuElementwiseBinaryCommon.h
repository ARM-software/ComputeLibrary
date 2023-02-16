/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_OPERATORS_INTERNAL_GPUELEMENTWISEBINARYCOMMON
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_OPERATORS_INTERNAL_GPUELEMENTWISEBINARYCOMMON

#include "arm_compute/core/Error.h"

namespace arm_compute
{
/** Forward declaration */
class ITensorInfo;

namespace experimental
{
namespace dynamic_fusion
{
class ElementwiseBinaryCommonAttributes
{
public:
    enum class ElementwiseOp
    {
        Add,         /**< (x + y) */
        Sub,         /**< (x - y) */
        Div,         /**< (x / y) */
        Mul,         /**< (x * y) */
        Min,         /**< Min(x, y) */
        Max,         /**< Max(x, y) */
        SquaredDiff, /**< (x - y)^2 */
        Power,       /**< x ^ y */
        Prelu,       /**< y*x if x < 0, x otherwise */
    };
    /** Set operation*/
    ElementwiseBinaryCommonAttributes &operation(const ElementwiseBinaryCommonAttributes::ElementwiseOp &operation);
    /** Get operation*/
    ElementwiseOp operation() const;

private:
    ElementwiseOp _operation; /**< Elementwise operation */
};

/** Forward declaration */
class GpuWorkloadContext;
class GpuWorkloadSketch;

/** Operator interface. */
class GpuElementwiseBinaryCommon final
{
public:
    /** Create an operator and fuse it into the workload sketch.
     *    @note If @ref validate_op() fails, the creation also fails and may throw an error.
     *    @note If @ref validate_op() fails, @p sketch remains unchanged and valid.
     *
     * Valid data type configurations are checked at the operator level i.e. GpuAdd::validate_op(), GpuSub::validate_op(), ... etc.
     *
     * Valid data layouts:
     * - Any
     *
     * @param[in,out] sketch     Workload sketch into which the operator will be fused
     * @param[in]     lhs        Left hand side tensor info. Data types supported: U8/S16/S32/F16/F32.
     * @param[in]     rhs        Right hand side tensor info. Data types supported: U8/S16/S32/F16/F32.
     * @param[in]     attributes ElementwiseBinaryCommonAttributes containing the operator type: ADD, SUB, DIV, ... etc.
     *
     * @return Pointer for the destination tensor info
     */
    static ITensorInfo *create_op(GpuWorkloadSketch                       &sketch,
                                  ITensorInfo                             *lhs,
                                  ITensorInfo                             *rhs,
                                  const ElementwiseBinaryCommonAttributes &attributes);
    /** Check if the operator configuration is supported, irrespective of fusion
     *
     * @param[in] context    Workload context within which the operator is running
     * @param[in] lhs        Left hand side tensor info. Data types supported: U8/S16/S32/F16/F32.
     * @param[in] rhs        Right hand side tensor info. Data types supported: U8/S16/S32/F16/F32.
     * @param[in] attributes ElementwiseBinaryCommonAttributes containing the operator type: ADD, SUB, DIV, ... etc.
     *
     * @return Status
     */
    static Status is_supported_op(const GpuWorkloadContext                &context,
                                  const ITensorInfo                       *lhs,
                                  const ITensorInfo                       *rhs,
                                  const ElementwiseBinaryCommonAttributes &attributes);
    /** Validate the operator and check if it can be fused into the workload sketch.
     *
     * Parameters are similar to @ref GpuElementwiseBinaryCommon::create_op()
     *
     * @return Status
     */
    static Status validate_op(const GpuWorkloadSketch                 &sketch,
                              const ITensorInfo                       *rhs,
                              const ITensorInfo                       *lhs,
                              const ElementwiseBinaryCommonAttributes &attributes);
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_OPERATORS_INTERNAL_GPUELEMENTWISEBINARYCOMMON */
