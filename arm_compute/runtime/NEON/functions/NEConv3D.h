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
#ifndef ARM_COMPUTE_NECONV3D_H
#define ARM_COMPUTE_NECONV3D_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/FunctionDescriptors.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Basic function to simulate a 3d convolution. This function calls one of the following functions:
 * -# @ref cpu::CpuDirectConv3d
 *
 */
class NEConv3D : public IFunction
{
public:
    /** Constructor */
    NEConv3D();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEConv3D(const NEConv3D &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEConv3D &operator=(const NEConv3D &) = delete;
    /** Default move constructor */
    NEConv3D(NEConv3D &&) = default;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEConv3D &operator=(NEConv3D &&) = default;
    /** Default destructor */
    ~NEConv3D();
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NDHWC
     *
     * Valid data type configurations:
     * |src0           |src1               |src2   |dst            |
     * |:--------------|:------------------|:------|:--------------|
     * |F16            |F16                |F16    |F16            |
     * |F32            |F32                |F32    |F32            |
     * |QASYMM8        |QASYMM8            |S32    |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32    |QASYMM8_SIGNED |
     *
     * @param[in]  input     Source tensor. 4 lower dimensions represent a single input [IFM, width, height, depth],
     *                       while every optional dimension from 5 and above represent a batch of inputs.
     * @param[in]  weights   Weights tensor. Weights are 5D tensor with dimensions [OFM, IFM, kernel_x, kernel_y, kernel_z].
     * @param[in]  biases    Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     * @param[out] output    Destination tensor. 4 lower dimensions represent a single output [OFM, width, height, depth], while the rest represent batch of outputs.
     * @param[in]  conv_info Contains padding, stride, acitvation information described in @ref Conv3dInfo.
     */
    void configure(ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const Conv3dInfo &conv_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to NEConv3D::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const Conv3dInfo &conv_info);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NECONV3D_H */
