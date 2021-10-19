/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CL_DIRECT_CONV3D_H
#define ARM_COMPUTE_CL_DIRECT_CONV3D_H

#include "src/gpu/cl/IClKernel.h"
#include "src/gpu/cl/IClOperator.h"

#include <memory>

namespace arm_compute
{
class CLCompileContext;
struct Conv3dInfo;
class IClKernel;

namespace opencl
{
/** Basic function to simulate a directly convolution layer with 3 spatial dimensions. This function calls the following OpenCL kernels:
 *
 * -# @ref opencl::ClDirectConv3d
 */
class ClDirectConv3d : public IClOperator
{
public:
    ClDirectConv3d() = default;
    /** Set the src and dst tensors.
     *
     * Valid data layouts:
     * - NDHWC
     *
     * Valid data type configurations:
     * |src0           |src1           |src2   |dst            |
     * |:--------------|:--------------|:------|:--------------|
     * |F16            |F16            |F16    |F16            |
     * |F32            |F32            |F32    |F32            |
     * |QASYMM8        |QASYMM8        |S32    |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |S32    |QASYMM8_SIGNED |
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src0            Source tensor. 4 lower dimensions represent a single src [IFM, width, height, depth],
     *                             while every optional dimension from 5 and above represent a batch of srcs.
     * @param[in]  src1            Weights tensor. Weights are 5D tensor with dimensions [OFM, IFM, kernel_w, kernel_h, kernel_d].
     * @param[in]  src2            Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     * @param[out] dst             Destination tensor. 4 lower dimensions represent a single dst [OFM, width, height, depth], while the rest represent batch of dsts.
     * @param[in]  conv3d_info     Contains strides, padding, rounding, activation, dilation and fast math information. Activation and fast math are currently unused.
     *
     */
    void configure(const CLCompileContext &compile_context, const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, ITensorInfo *dst, const Conv3dInfo &conv3d_info);

    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to ClDirectConv3d::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, const Conv3dInfo &conv3d_info);

    // Inherited method overridden
    void run(ITensorPack &tensors) override;

private:
    std::unique_ptr<IClKernel> _direct_conv3d_kernel{ nullptr };
};
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_DIRECT_CONV3D_H */