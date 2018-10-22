/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLGEMMLOWPQUANTIZEDOWNINT32TOUINT8SCALEBYFLOATKERNEL_H__
#define __ARM_COMPUTE_CLGEMMLOWPQUANTIZEDOWNINT32TOUINT8SCALEBYFLOATKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
// Forward declarations
class ICLTensor;

/** OpenCL kernel used to quantize down the int32 accumulator values of GEMMLowp to QASYMM8
 *
 * This kernel takes a final int32 accumulator value (the output of @ref CLGEMMLowpMatrixMultiplyKernel), and processes it to obtain the final QASYMM8 value.
 * The following computations will be performed by the kernel:
 *
 *  -# Compute fixed point multiplication between each entry of input by result_fixedpoint_multiplier
 *  -# Add bias to final result if bias tensor is not a nullptr
 *  -# Requantize
 *  -# Add offset to each result
 *  -# Clamp the value between the specified min and max bounds
 *  -# Clamp the resulting int32 values to the [0..255] range and cast to QASYMM8.
 */
class CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFloatKernel : public ICLKernel
{
public:
    /** Constructor */
    CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFloatKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers)*/
    CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFloatKernel(const CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFloatKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers)*/
    CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFloatKernel &operator=(const CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFloatKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFloatKernel(CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFloatKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFloatKernel &operator=(CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFloatKernel &&) = default;
    /** Initialise the kernel's input and output.
     *
     * @param[in]  input           Input tensor. Data type supported: S32
     * @param[in]  bias            Biases tensor. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                             Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output          Output tensor. Data type supported: Data type supported: QASYMM8
     * @param[in]  multiplier      Float multiplier to be multiplied to each element of the input matrix
     * @param[in]  offset          Offset to be applied to result before converting it back to QASYMM8
     * @param[in]  min             (Optional) Min value used to saturate down the output result before converting back to QASYMM8
     * @param[in]  max             (Optional) Max value used to saturate up the output result before converting back to QASYMM8,
     *                             Along with @p min, this value can be used to implement "rectified linear unit" activation functions
     * @param[in]  output_3d_depth (Optional) Depth of output in 3D (Defaults to 1)
     */
    void configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, float multiplier, int offset,
                   int min = 0, int max = 0, unsigned int output_3d_depth = 1);
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPointKernel
     *
     * @param[in] input           Input tensor. Data type supported: S32
     * @param[in] bias            Biases tensor. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                            Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[in] output          Output tensor. Data type supported: Data type supported: QASYMM8
     * @param[in] min             (Optional) Min value used to saturate down the output result before converting back to QASYMM8
     * @param[in] max             (Optional) Max value used to saturate up the output result before converting back to QASYMM8,
     *                            Along with @p min, this value can be used to implement "rectified linear unit" activation functions
     * @param[in] output_3d_depth (Optional)  Depth of output in 3D (Defaults to 1)
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output,
                           int min = 0, int max = 0, unsigned int output_3d_depth = 1);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    const ICLTensor *_bias;
    ICLTensor       *_output;
    bool             _reinterpret_as_3d;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLGEMMLOWPQUANTIZEDOWNINT32TOUINT8SCALEBYFLOATKERNEL_H__ */
