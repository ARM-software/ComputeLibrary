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
#ifndef __ARM_COMPUTE_NEGEMMLOWPQUANTIZEDOWNINT32TOUINT8SCALEKERNEL_H__
#define __ARM_COMPUTE_NEGEMMLOWPQUANTIZEDOWNINT32TOUINT8SCALEKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** NEON kernel used to quantize down the int32 accumulator values of GEMMLowp to QASYMM8
 *
 * This kernel takes a final int32 accumulator value (the output of @ref NEGEMMLowpMatrixMultiplyKernel), and processes it to obtain the final QASYMM8 value.
 * The following computations will be performed by the kernel:
 *
 *  -# Add offset terms to final result
 *  -# Multiply each entry of result by result_mult_int
 *  -# Add bias to final result if bias tensor is not a nullptr
 *  -# Shift the int32 accumulator by result_shift
 *  -# Clamp the value between the specified min and max bounds
 *  -# Clamp the resulting int32 values to the [0..255] range and cast to QASYMM8.
 *
 */
class NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel : public INEKernel
{
public:
    /** Constructor */
    NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers)*/
    NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel(const NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers)*/
    NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel &operator=(const NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel(NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel &operator=(NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel &&) = default;
    /** Initialise the kernel's input and output.
     *
     * @param[in]  input           Input tensor. Data type supported: S32
     * @param[in]  bias            Biases tensor. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                             Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output          Output tensor. Data type supported: Data type supported: QASYMM8
     * @param[in]  result_offset   Offset to be added to each element of the input matrix
     * @param[in]  result_mult_int Value to be multiplied to each element of the input matrix when once the result_offset has been add
     * @param[in]  result_shift    Number of bits to shift right the result before converting back to QASYMM8
     * @param[in]  min             (Optional) Min value used to saturate down the output result before converting back to QASYMM8
     * @param[in]  max             (Optional) Max value used to saturate up the output result before converting back to QASYMM8,
     *                             Along with @p min, this value can be used to implement "rectified linear unit" activation functions
     */
    void configure(const ITensor *input, const ITensor *bias, ITensor *output, int result_offset, int result_mult_int, int result_shift, int min = 0, int max = 0);
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel
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
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Template function to run the NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel
     *
     * @param[in] window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    template <bool is_bounded_relu>
    void run(const Window &window);

    /** Common signature for all the specialised NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using QuantizeDownFunctionPtr = void (NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel::*)(const Window &window);

    QuantizeDownFunctionPtr _func;
    const ITensor          *_input;
    const ITensor          *_bias;
    ITensor                *_output;
    int                     _result_offset;
    int                     _result_mult_int;
    int                     _result_shift;
    int                     _min;
    int                     _max;
};
} // namespace arm_compute

#endif /* __ARM_COMPUTE_NEGEMMLOWPQUANTIZEDOWNINT32TOUINT8SCALEKERNEL_H__ */
