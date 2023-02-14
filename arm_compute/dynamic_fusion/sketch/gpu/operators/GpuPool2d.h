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
#ifndef ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_GPU_OPERATORS_GPUPOOL2D
#define ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_GPU_OPERATORS_GPUPOOL2D

#include "arm_compute/core/Error.h"
#include "arm_compute/dynamic_fusion/sketch/attributes/Pool2dAttributes.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Forward declaration */
class GpuWorkloadSketch;
class GpuWorkloadContext;

/** Operator backend specific settings
*/
class GpuPool2dSettings
{
public:
    /* Get mixed_precision*/
    bool mixed_precision() const;

    /* Set mixed_precision */
    GpuPool2dSettings &mixed_precision(bool mixed_precision);

    /* Get using -infinity as limit flag */
    bool use_inf_as_limit() const;

    /* Set using -infinity as limit flag */
    GpuPool2dSettings use_inf_as_limit(bool use_inf_as_limit);

private:
    bool _mixed_precision{ false };
    bool _use_inf_as_limit{ true };
};

/** Operator interface. */
class GpuPool2d final
{
public:
    /** Attributes are a set of backend-agnostic parameters that define what an operator does */
    using Attributes = Pool2dAttributes;
    /** Settings are a set of backend-specific parameters that influence the implementation of a operator */
    using Settings = GpuPool2dSettings;

    /** Create an operator and fuse it into the workload sketch.
     *    @note If @ref validate_op() fails, the creation also fails and may throw an error.
     *    @note If @ref validate_op() fails, @p sketch remains unchanged and valid.
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |F16            |F16            |
     * |F32            |F32            |
     *
     * Valid data layouts:
     * - NHWC
     *
     * @param[in,out] sketch     Workload sketch into which the operator will be fused
     * @param[in]     src        Source tensor
     * @param[out]    dst        Destination tensor
     * @param[in]     attributes Operator attributes
     * @param[in]     settings   Operator settings
     */
    static void create_op(GpuWorkloadSketch &sketch,
                          ITensorInfo       *src,
                          ITensorInfo       *dst,
                          const Attributes &attributes,
                          const Settings    &settings);
    /** Check if the operator configuration is supported, irrespective of fusion
     *
     * @param[in]  context    Workload context within which the operator is running
     * @param[in]  src        Left hand side tensor info. Data types supported: F16/F32.
     * @param[out] dst        Destination tensor info. Data types supported: F16/F32.
     *                        If an uninitialized ITensorInfo is passed in, it will be auto-initialized
     * @param[in]  attributes Operator attributes
     * @param[in]  settings   Operator settings
     */
    static Status is_supported_op(const GpuWorkloadContext &context,
                                  const ITensorInfo        *src,
                                  const ITensorInfo        *dst,
                                  const Attributes         &attributes,
                                  const Settings           &settings);
    /** Validate the operator and check if it can be fused into the workload sketch.
     * Similar to @ref GpuPool2d::create_op()
     */
    static Status validate_op(const GpuWorkloadSketch &sketch,
                              const ITensorInfo       *src,
                              const ITensorInfo       *dst,
                              const Attributes        &attributes,
                              const Settings          &settings);
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_GPU_OPERATORS_GPUPOOL2D */
