/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NEGEMMCONV2D_H
#define ARM_COMPUTE_NEGEMMCONV2D_H

#include "arm_compute/runtime/FunctionDescriptors.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;

/** Basic function to compute the convolution layer. This function calls the following kernels/functions:
 *
 * Supports only NHWC data layout
 *
 * -# @ref cpu::CpuGemmAssemblyDispatch
 * -# @ref NEActivationLayer, in case activation cannot be fused in the assembly dispatch
 *
 * Weights are transformed from OHWI to HWIO format using the following kernels:
 * -# @ref NEPermute
 */
class NEGEMMConv2d : public IFunction
{
public:
    /** Constructor */
    NEGEMMConv2d(const std::shared_ptr<IMemoryManager> &memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMConv2d(const NEGEMMConv2d &) = delete;
    /** Default move constructor */
    NEGEMMConv2d(NEGEMMConv2d &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMConv2d &operator=(const NEGEMMConv2d &) = delete;
    /** Default move assignment operator */
    NEGEMMConv2d &operator=(NEGEMMConv2d &&) = default;
    /** Destructor */
    ~NEGEMMConv2d();
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |src1           |src2           |dst            |
     * |:--------------|:--------------|:--------------|:--------------|
     * |QASYMM8        |QASYMM8        |S32            |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |S32            |QASYMM8_SIGNED |
     * |F16            |F16            |F16            |F16            |
     * |F32            |F32            |F32            |F32            |
     * |BFLOAT16       |BFLOAT16       |BFLOAT16       |BFLOAT16       |
     *
     * @param[in]  input   Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                     while every optional dimension from 4 and above represent a batch of inputs.
     *                     Data types supported: QASYMM8/QASYMM8_SIGNED/BFLOAT16/F16/F32.
     * @param[in]  weights Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                     Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL/BFLOAT16/F16/F32.
     * @param[in]  biases  Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                     Data type supported: Should match @p input data type, except for input of QASYMM8/QASYMM8_SIGNED type where biases should be of S32 type.
     * @param[out] output  Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                     Data types supported: Same as @p input.
     * @param[in]  info    Convolution layer descriptor
     */
    void configure(ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const Conv2dInfo &info);
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMConv2d
     *
     * @param[in] input   Source tensor info. 3 lower dimensions represent a single input [width, height, IFM],
     *                    while every optional dimension from 4 and above represent a batch of inputs.
     *                    Data types supported: QASYMM8/QASYMM8_SIGNED/BFLOAT16/F16/F32.
     * @param[in] weights Weights tensor info. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                    Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL/BFLOAT16/F16/F32.
     * @param[in] biases  Biases tensor info. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                    Data type supported: Should match @p input data type, except for input of QASYMM8/QASYMM8_SIGNED type where biases should be of S32 type.
     * @param[in] output  Destination tensor info. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                    Data types supported: Same as @p input.
     * @param[in] info    Contains padding and stride information described in @ref PadStrideInfo.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const Conv2dInfo &info);

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEGEMMCONV2D_H */
