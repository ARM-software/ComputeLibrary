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
#ifndef ARM_COMPUTE_CLCONV2D_H
#define ARM_COMPUTE_CLCONV2D_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/FunctionDescriptors.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"
#include "src/gpu/cl/IClOperator.h"

namespace arm_compute
{
namespace opencl
{
/** Basic function to compute the convolution layer. This function calls the following OpenCL kernels/functions:
 *
 * -# @ref opencl::ClGemmConv2d
 * -# @ref opencl::ClWinogradConv2d
 * -# @ref opencl::ClIndirectConv2d
 * -# @ref opencl::ClDirectConv2d
 * -# @ref CLFFTConvolutionLayer
 *
 * The function selects one of the algorithms mentioned above based on:
 *      - The size of the kernel
 *      - Number of src/dst feature maps
 *      - Amount of memory needed
 *
 * Generally GEMM-based convolution is executed when neither Winograd nor FFT nor Direct convolution can be performed.
 *
 * FP32 Algorithm| Filter Size                                                 |   Input/Output feature maps               |
 * --------------|-------------------------------------------------------------|-------------------------------------------|
 * Winograd      | 3x3 1x3 3x1 5x1 1x5 5x5(fast maths) 7x1 1x7                 |  Input channels is greater than 3         |
 * FFT           | Squared kernels and greater than 9x9                        |  Input feature maps > Output feature maps |
 * DirectConv    | 9x9                                                         |                                           |
 * GEMM          | Any size                                                    |                                           |
 *
 * Winograd 5x5 requires fast maths enabled.
 *
 * FP16 Algorithm| Filter Size                |   Input/Output feature maps               |
 * --------------|----------------------------|-------------------------------------------|
 * Winograd      | 3x3 1x3 3x1 5x1 1x5 5x5    |  Input channels is greater than 3         |
 * FFT           | Not supported              |                                           |
 * DirectConv    | 9x9                        |                                           |
 * GEMM          | Any size                   |                                           |
 *
 * Winograd FP16 requires fast maths enabled.
 *
 */
class ClConv2d : public IClOperator
{
public:
    /** Default constructor */
    ClConv2d();
    /** Default Destructor */
    ~ClConv2d();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    ClConv2d(const ClConv2d &) = delete;
    /** Default move constructor */
    ClConv2d(ClConv2d &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    ClConv2d &operator=(const ClConv2d &) = delete;
    /** Default move assignment operator */
    ClConv2d &operator=(ClConv2d &&) = default;
    /** Set the src and dst tensors.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src0           |src1               |src2   |dst            |
     * |:--------------|:------------------|:------|:--------------|
     * |F16            |F16                |F16    |F16            |
     * |F32            |F32                |F32    |F32            |
     * |QASYMM8        |QASYMM8            |S32    |QASYMM8        |
     * |QASYMM8        |QSYMM8_PER_CHANNEL |S32    |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32    |QASYMM8_SIGNED |
     * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32    |QASYMM8_SIGNED |
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             Source tensor info. 3 lower dimensions represent a single src [width, height, IFM],
     *                             while every optional dimension from 4 and above represent a batch of srcs.
     *                             Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  weights         Weights tensor info. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                             Data type supported: Same as @p src, also could be QSYMM8_PER_CHANNEL if src is QASYMM8/QASYMM8_SIGNED.
     * @param[in]  biases          Biases tensor info. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                             Data type supported: Same as @p src, except for src of QASYMM8/QASYMM8_SIGNED type where biases should be of S32 type.
     * @param[out] dst             Destination tensor info. 3 lower dimensions represent a single dst [width, height, OFM], while the rest represent batch of dsts.
     *                             Data types supported: Same as @p src.
     * @param[in]  conv2d_info     Contains convolution 2d info described in @ref Conv2dInfo.
     * @param[in]  weights_info    Specifies if the weights tensor has been reshaped with CLWeightsReshapeKernel. Data type supported: Same as @p src.
     */
    void configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *weights, ITensorInfo *biases, ITensorInfo *dst, const Conv2dInfo &conv2d_info,
                   const WeightsInfo &weights_info = WeightsInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref ClConv2d
     *
     * Similar to ClConv2d::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const Conv2dInfo &conv2d_info,
                           const WeightsInfo &weights_info = WeightsInfo());
    /** Static function to check if given info will return the convolution called by @ref ClConv2d
     *
     * @param[in] src          Source tensor. 3 lower dimensions represent a single src [width, height, IFM],
     *                         while every optional dimension from 4 and above represent a batch of srcs.
     *                         Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] weights      Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                         Data type supported: Same as @p src, also could be QSYMM8_PER_CHANNEL if src is QASYMM8/QASYMM8_SIGNED.
     * @param[in] dst          Destination tensor. 3 lower dimensions represent a single dst [width, height, OFM], while the rest represent batch of dsts.
     *                         Data types supported: Same as @p src.
     * @param[in] conv2d_info  Contains convolution 2d info described in @ref Conv2dInfo.
     * @param[in] weights_info Specifies if the weights tensor has been reshaped with CLWeightsReshapeKernel.
     * @param[in] gpu_target   Specifies the @p GPUTarget.
     *
     * @return the Convolution Method Hint
     */
    static ConvolutionMethod get_convolution_method(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *dst, const Conv2dInfo &conv2d_info,
                                                    const WeightsInfo &weights_info, const GPUTarget gpu_target);
    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
    void prepare(ITensorPack &tensors) override;
    experimental::MemoryRequirements workspace() const override;

private:
    std::unique_ptr<IClOperator>     _operator;
    experimental::MemoryRequirements _aux_mem{};
};
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLCONV2D_H */
