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
#ifndef __ARM_COMPUTE_NEGEMMLOWPQUANTIZEDOWNINT32TOUINT8SCALE_H__
#define __ARM_COMPUTE_NEGEMMLOWPQUANTIZEDOWNINT32TOUINT8SCALE_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/* NEON kernel used to quantize down the int32 accumulator values of GEMMLowp to QASYMM8
 *
 * This kernel takes a final int32 accumulator value (the output of @ref NEGEMMLowpMatrixMultiplyKernel), and processes it to obtain the final QASYMM8 value.
 * The following computations will be performed by the kernel:
 *
 *  -# Add offset terms to final result
 *  -# Multiply each entry of result and round to nearest integer
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
    * @param[out] output          Output tensor. Data type supported: Data type supported: QASYMM8
    * @param[in]  result_offset   Offset to be added to each element of the input matrix
    * @param[in]  result_mult_int Value to be multiplied to each element of the input matrix when once the result_offset has been add
    * @param[in]  result_shift    Number of bits to shift right the result before converting back to QASYMM8
     */
    void configure(const ITensor *input, ITensor *output, int result_offset, int result_mult_int, int result_shift);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor *_input;
    ITensor       *_output;
    int32_t        _result_offset;
    int32_t        _result_mult_int;
    int32_t        _result_shift;
};
} // namespace arm_compute

#endif /* __ARM_COMPUTE_NEGEMMLOWPQUANTIZEDOWNINT32TOUINT8SCALE_H__ */
