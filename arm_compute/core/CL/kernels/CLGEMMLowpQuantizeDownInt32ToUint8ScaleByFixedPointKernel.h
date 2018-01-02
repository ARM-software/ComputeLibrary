/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLGEMMLOWPQUANTIZEDOWNINT32TOUINT8SCALEBYFIXEDPOINTKERNEL_H__
#define __ARM_COMPUTE_CLGEMMLOWPQUANTIZEDOWNINT32TOUINT8SCALEBYFIXEDPOINTKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel used to quantize down the int32 accumulator values of GEMMLowp to QASYMM8
 *
 * This kernel takes a final int32 accumulator value (the output of @ref CLGEMMLowpMatrixMultiplyKernel), and processes it to obtain the final QASYMM8 value.
 * The following computations will be performed by the kernel:
 *
 *  -# Compute fixed point multiplication between each entry of input by result_fixedpoint_multiplier
 *  -# Add bias to final result if bias tensor is not a nullptr
 *  -# Round to nearest division by a power-of-two using result_shift
 *  -# Add offset to each result
 *  -# Clamp the value between the specified min and max bounds
 *  -# Clamp the resulting int32 values to the [0..255] range and cast to QASYMM8.
 */
class CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel : public ICLKernel
{
public:
    /** Constructor */
    CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers)*/
    CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel(const CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers)*/
    CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel &operator=(const CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel(CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel &operator=(CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel &&) = default;
    /** Initialise the kernel's input and output.
     *
     * @param[in]  input                        Input tensor. Data type supported: S32
     * @param[in]  bias                         Biases tensor. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                                          Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output                       Output tensor. Data type supported: Data type supported: QASYMM8
     * @param[in]  result_fixedpoint_multiplier Fixed point value to be multiplied to each element of the input matrix when once the result_offset has been add
     * @param[in]  result_shift                 Integer value used to round to nearest division by a power-of-two the result after the fixed point multiplication
     * @param[in]  result_offset_after_shift    Offset to be applied to result before converting it back to QASYMM8
     * @param[in]  min                          (Optional) Min value used to saturate down the output result before converting back to QASYMM8
     * @param[in]  max                          (Optional) Max value used to saturate up the output result before converting back to QASYMM8,
     *                                          Along with @p min, this value can be used to implement "rectified linear unit" activation functions
     */
    void configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift, int min = 0, int max = 0);
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel
     *
     * @param[in] input  Input tensor. Data type supported: S32
     * @param[in] bias   Biases tensor. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                   Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[in] output Output tensor. Data type supported: Data type supported: QASYMM8
     * @param[in] min    (Optional) Min value used to saturate down the output result before converting back to QASYMM8
     * @param[in] max    (Optional) Max value used to saturate up the output result before converting back to QASYMM8,
     *                   Along with @p min, this value can be used to implement "rectified linear unit" activation functions
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, int min = 0, int max = 0);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    const ICLTensor *_bias;
    ICLTensor       *_output;
};
} // namespace arm_compute

#endif /* __ARM_COMPUTE_CLGEMMLOWPQUANTIZEDOWNINT32TOUINT8SCALEBYFIXEDPOINTKERNEL_H__ */
