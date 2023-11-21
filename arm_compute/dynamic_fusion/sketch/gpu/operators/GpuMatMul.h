/*
 * Copyright (c) 2023 Arm Limited.
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
#ifndef ACL_ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_GPU_OPERATORS_GPUMATMUL_H
#define ACL_ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_GPU_OPERATORS_GPUMATMUL_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/MatMulAttributes.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Forward declaration */
class GpuWorkloadContext;
class GpuWorkloadSketch;

/** Operator backend specific settings
*/
class GpuMatMulSettings
{
public:
    /** Set N0: number of columns processed by each work-item */
    GpuMatMulSettings &n0(int n0);
    /** Get N0: number of columns processed by each work-item */
    int n0() const;

    /** Set M0: number of rows processed by each work-item */
    GpuMatMulSettings &m0(int m0);
    /** Get M0: number of rows processed by each work-item */
    int m0() const;

    /** Set K0: number of inner accumulations */
    GpuMatMulSettings &k0(int k0);
    /** Get K0: number of inner accumulations */
    int k0() const;

private:
    int _n0{0}; /**< Number of columns processed by each work-item */
    int _m0{0}; /**< Number of rows processed by each work-item */
    int _k0{0}; /**< Number of inner accumulations */
};

/** Operator interface. */
class GpuMatMul final
{
public:
    /* Attributes are a set of backend-agnostic parameters that define what an operator does */
    using Attributes = MatMulAttributes;
    /* Settings are a set of backend-specific parameters that influence the implementation of a operator */
    using Settings = GpuMatMulSettings;

    /* Create an operator and fuse it into the workload sketch.
     *    @note If @ref validate_op() fails, the creation also fails and may throw an error.
     *    @note If @ref validate_op() fails, @p sketch remains unchanged and valid.
     *
     * Valid data type configurations:
     * |LHS            |RHS            |dst            |
     * |:--------------|:--------------|:--------------|
     * |F16            |F16            |F16            |
     * |F32            |F32            |F32            |
     *
     *
     * @param[in,out] sketch     Workload sketch into which the operator will be fused
     * @param[in]     lhs        Input tensor info for the LHS matrix. Data type supported: F32/F16. Dimensions above 2 are collapsed onto dimension 2 and represent the batch.
     * @param[in]     rhs        Input tensor info for the RHS matrix. Data type supported: same as @p lhs. Dimensions above 2 are collapsed onto dimension 2 and represent the batch.
     * @param[in]     attributes Operator attributes
     * @param[in]     settings   Operator settings
     *
     * @return Pointer for the destination tensor info
     */
    static ITensorInfo *create_op(GpuWorkloadSketch &sketch,
                                  ITensorInfo       *lhs,
                                  ITensorInfo       *rhs,
                                  const Attributes  &attributes,
                                  const Settings    &settings);

    /* Check if the operator configuration is supported, irrespective of fusion
     *
     * @param[in] context    Workload context within which the operator is running
     * @param[in] lhs        Input tensor info for the LHS matrix.
     * @param[in] rhs        Input tensor info for the RHS matrix.
     * @param[in] attributes Operator attributes
     * @param[in] settings   Operator settings
     *
     * @return Status
     */
    static Status is_supported_op(const GpuWorkloadContext &context,
                                  const ITensorInfo        *lhs,
                                  const ITensorInfo        *rhs,
                                  const Attributes         &attributes,
                                  const Settings           &settings);

    /* Check if the operator configuration is supported and if it can be fused into the workload sketch.
     *
     * Parameters are similar to @ref GpuMatMul::create_op()
     *
     * @return Status
     */
    static Status validate_op(const GpuWorkloadSketch &sketch,
                              const ITensorInfo       *lhs,
                              const ITensorInfo       *rhs,
                              const Attributes        &attributes,
                              const Settings          &settings);
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif // ACL_ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_GPU_OPERATORS_GPUMATMUL_H
