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
#ifndef ARM_COMPUTE_CL_TRANSPOSED_CONVOLUTION_H
#define ARM_COMPUTE_CL_TRANSPOSED_CONVOLUTION_H

#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"
#include "src/gpu/cl/IClOperator.h"
namespace arm_compute
{
namespace opencl
{
/** Basic function to simulate a directly convolution layer. This function calls the following OpenCL kernels:
 *
 * -# @ref opencl::ClTransposedConvolution
 */
class ClTransposedConvolution : public IClOperator
{
public:
    /** Default constructor */
    ClTransposedConvolution() = default;
    /** Default Destructor */
    ~ClTransposedConvolution() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    ClTransposedConvolution(const ClTransposedConvolution &) = delete;
    /** Default move constructor */
    ClTransposedConvolution(ClTransposedConvolution &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    ClTransposedConvolution &operator=(const ClTransposedConvolution &) = delete;
    /** Default move assignment operator */
    ClTransposedConvolution &operator=(ClTransposedConvolution &&) = default;

    /** Set the input, weights, biases and output tensors.
     *
     * @note: Only NHWC data layout is supported
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Input tensor info with dimensions [IFM, width, height, batch]
     *                             Data types supported: F16/F32/QASYMM8/QASYMM8_SIGNED.
     * @param[in]  weights         Weight tensor info with dimensions [IFM, width, height, OFM].
     *                             Data type supported: Same as @p input
     * @param[in]  biases          (Optional) Biases tensor info. Biases are 1D tensor with dimension [OFM].
     *                             Data type supported: Should match @p input data type if floating point, otherwise S32.
     * @param[out] output          Output tensor info with dimensions [OFM, width, height, batch]
     *                             The 1st dimension must be equal to the 4th dimension of the @p weights tensor.
     *                             Data types supported: Same as @p input.
     * @param[in]  deconv_info     Contains padding and stride information described in @ref PadStrideInfo.
     *
     */
    void configure(const CLCompileContext &compile_context, const ITensorInfo *input, const ITensorInfo *weights,
                   const ITensorInfo *biases, ITensorInfo *output, const PadStrideInfo &deconv_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to ClTransposedConvolution::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases,
                           const ITensorInfo *output, const PadStrideInfo &deconv_info);

    // Inherited method overridden
    void run(ITensorPack &tensors) override;

private:
    std::unique_ptr<IClKernel> _transposed_conv_kernel{ nullptr };
};
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_TRANSPOSED_CONVOLUTION_H */