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
#ifndef ARM_COMPUTE_CL_INDIRECT_CONV2D_H
#define ARM_COMPUTE_CL_INDIRECT_CONV2D_H

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTypes.h"

#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"
#include "src/gpu/cl/IClOperator.h"

#include <memory>

namespace arm_compute
{
// Forward declaration
struct DirectConvComputeKernelInfo;

namespace opencl
{
/** Basic function to execute indirect convolution on OpenCL. This function calls the following OpenCL kernels:
 *
 *  -# @ref kernels::ClIndirectConv2dAddressPrecalculationKernel
 *  -# @ref kernels::ClIndirectConv2dKernel
 */
class ClIndirectConv2d : public IClOperator
{
public:
    ClIndirectConv2d() = default;
    /** Initialise the kernel's inputs and output
     *
     * Valid data layouts:
     * - NHWC
     *
     * Valid data type configurations:
     * |src0         |src1        |src2      |dst            |
     * |:------------|:-----------|:---------|:--------------|
     * |F32          |F32         |F32       |F32            |
     * |F16          |F16         |F16       |F16            |
     *
     * @note All tensors must have the same data type.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             Source tensor. 3 lower dimensions represent a single src,
     *                             while every optional dimension from 4 and above represent a batch of sources.
     *                             Data types supported: F16/F32.
     * @param[in]  weights         Weights tensor. Weights are 4D tensor with dimensions. Data type supported:Same as @p src.
     * @param[in]  biases          Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                             Data type supported: Should match @p src data type.
     * @param[out] dst             Destination tensor. 3 lower dimensions represent a single dst, while the rest represent batch of destinations.
     *                             Data types supported: Same as @p src.
     * @param[in]  conv_info       Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  act_info        (Optional) Activation layer information in case of a fused activation.
     *
     */
    void configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *weights, ITensorInfo *biases, ITensorInfo *dst, const PadStrideInfo &conv_info,
                   const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to ClIndirectConv2d::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const PadStrideInfo &conv_info,
                           const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
    void prepare(ITensorPack &constants) override;
    experimental::MemoryRequirements workspace() const override;

private:
    enum AuxTensorIdx
    {
        IndirectBuffer = 0,
        Count
    };

    std::unique_ptr<IClKernel>       _indirect_conv_kernel{ nullptr };
    std::unique_ptr<IClKernel>       _addr_precalculation_kernel{ nullptr };
    TensorInfo                       _indirect_buffer{};
    bool                             _is_prepared{ false };
    experimental::MemoryRequirements _aux_mem{ Count };
};
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_INDIRECT_CONV2D_H */
