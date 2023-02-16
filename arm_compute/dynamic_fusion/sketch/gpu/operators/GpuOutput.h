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

#ifndef ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_GPU_OPERATORS_GPUOUTPUT
#define ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_GPU_OPERATORS_GPUOUTPUT

#include "arm_compute/core/ITensorInfo.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Forward declaration */
class GpuWorkloadContext;
class GpuWorkloadSketch;

/** Operator interface. */
class GpuOutput final
{
public:
    /** Create an operator and fuse it into the workload sketch.
     *    @note If @ref validate_op() fails, the creation also fails and may throw an error.
     *    @note If @ref validate_op() fails, @p sketch remains unchanged and valid.
     *
     * Valid data type configurations:
     *   - Any
     *
     * Valid data layouts:
     *   - Any
     *
     * @param[in, out] sketch Workload sketch into which the operator will be fused.
     * @param[in, out] src    Source tensor info.
     * @param[in, out] dst    Destination tensor info.
     *                        If an uninitialized ITensorInfo is passed in, it will be auto-initialized.
     */
    static void create_op(GpuWorkloadSketch &sketch,
                          ITensorInfo       *src,
                          ITensorInfo       *dst);

    /** Check if the operator configuration is supported, irrespective of fusion.
     *
     * @param[in] context Workload context within which the operator is running.
     * @param[in] src     Source tensor info.
     * @param[in] dst     Destination tensor info.
     *
     * @return Status
     */
    static Status is_supported_op(const GpuWorkloadContext &context,
                                  const ITensorInfo        *src,
                                  const ITensorInfo        *dst);

    /** Validate the operator and check if the its configuration is supported and if it can be fused into the workload sketch.
     *
     * Parameters are similar to @ref GpuOutput::create_op().
     *
     * @return Status
     */
    static Status validate_op(const GpuWorkloadSketch &sketch,
                              const ITensorInfo       *src,
                              const ITensorInfo       *dst);
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif /* ARM_COMPUTE_DYNAMIC_FUSION_SKETCH_GPU_OPERATORS_GPUOUTPUT */
