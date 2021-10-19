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
#ifndef ARM_COMPUTE_CLCONVOLUTION3DLAYER_H
#define ARM_COMPUTE_CLCONVOLUTION3DLAYER_H

#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;
struct Conv3dInfo;
class Status;

/** Basic function to compute the convolution3d layer. This function calls the following OpenCL kernels/functions:
 *
 * -# @ref opencl::ClDirectConv3d
 */
class CLConv3D : public IFunction
{
public:
    /** Construtor */
    CLConv3D();
    /** Destructor */
    ~CLConv3D();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLConv3D(const CLConv3D &) = delete;
    /** Default move constructor */
    CLConv3D(CLConv3D &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLConv3D &operator=(const CLConv3D &) = delete;
    /** Default move assignment operator */
    CLConv3D &operator=(CLConv3D &&) = default;
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
     * @param[in]  src             Source tensor. 4 lower dimensions represent a single src [IFM, width, height, depth],
     *                             while every optional dimension from 5 and above represent a batch of srcs.
     * @param[in]  weights         Weights tensor. Weights are 5D tensor with dimensions [OFM, IFM, kernel_w, kernel_h, kernel_d].
     * @param[in]  biases          Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     * @param[out] dst             Destination tensor. 4 lower dimensions represent a single dst [OFM, width, height, depth], while the rest represent batch of dsts.
     * @param[in]  conv3d_info     Contains strides, padding, rounding, activation, dilation and fast math information. Activation and fast math are currently unused.
     *
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *src, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *dst, const Conv3dInfo &conv3d_info);
    /** Set the src and dst tensors.
     *
     * Similar to CLConv3D::configure() but using the default compile context
     *
     */
    void configure(const ICLTensor *src, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *dst, const Conv3dInfo &conv3d_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLConv3D
     *
     * Similar to CLConv3D::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const Conv3dInfo &conv3d_info);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
}
#endif /* ARM_COMPUTE_CLCONVOLUTION3DLAYER_H */
