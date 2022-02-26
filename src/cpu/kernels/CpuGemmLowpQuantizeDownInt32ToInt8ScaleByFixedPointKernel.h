/*
 * Copyright (c) 2019-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_GEMMLOWP_QUANTIZEDOWN_INT32TOINT8_SCALEBYFIXEDPOINT_KERNEL_H
#define ARM_COMPUTE_CPU_GEMMLOWP_QUANTIZEDOWN_INT32TOINT8_SCALEBYFIXEDPOINT_KERNEL_H

#include "arm_compute/core/KernelDescriptors.h"
#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
// Forward declaration
class ITensor;
namespace cpu
{
namespace kernels
{
/** Kernel used to quantize down the int32 accumulator values of GEMMLowp to QASYMM8_SIGNED
 *
 * This kernel takes a final int32 accumulator value (the output of @ref CpuGemmLowpMatrixMultiplyKernel), and processes it to obtain the final QASYMM8_SIGNED value.
 * The following computations will be performed by the kernel:
 *
 *  -# Compute fixed point multiplication between each entry of input by result_fixedpoint_multiplier
 *  -# Add bias to final result if bias tensor is not a nullptr
 *  -# Round to nearest division by a power-of-two using result_shift
 *  -# Add offset to each result
 *  -# Clamp the value between the specified min and max bounds
 *  -# Clamp the resulting int32 values to the [-128..127] range and cast to QASYMM8_SIGNED.
 *
 */
class CpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel : public ICpuKernel<CpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel>
{
public:
    CpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel);
    /** Initialise the kernel's input and output.
     *
     * @param[in]  src                          Input tensor info. Data type supported: S32
     * @param[in]  bias                         Biases tensor info. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                                          Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] dst                          Output tensor info. Data type supported: Data type supported: QASYMM8_SIGNED
     * @param[in]  result_fixedpoint_multiplier Fixed point value to be multiplied to each element of the input matrix when once the result_offset has been add
     * @param[in]  result_shift                 Integer value used to round to nearest division by a power-of-two the result after the fixed point multiplication
     * @param[in]  result_offset_after_shift    Offset to be applied to result before converting it back to QASYMM8_SIGNED
     * @param[in]  min                          (Optional) Min value used to saturate down the output result before converting back to QASYMM8_SIGNED
     * @param[in]  max                          (Optional) Max value used to saturate up the output result before converting back to QASYMM8_SIGNED,
     *                                          Along with @p min, this value can be used to implement "rectified linear unit" activation functions
     */
    void configure(ITensorInfo *src, ITensorInfo *bias, ITensorInfo *dst, int result_fixedpoint_multiplier, int result_shift, int result_offset_after_shift, int min = 0, int max = 0);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *bias, const ITensorInfo *dst, int min = 0, int max = 0);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

private:
    /** Template function to run the CpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel
     *
     * @param[in]  src    Input tensor info
     * @param[in]  bias   Bias tensor info
     * @param[out] dst    Output tensor info
     * @param[in]  window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    template <bool is_bounded_relu>
    void run_internal(const ITensor *src, const ITensor *bias, ITensor *dst, const Window &window);

    /** Common signature for all the specialised CpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel functions
     *
     * @param[in]  src    Input tensor info
     * @param[in]  bias   Bias tensor info
     * @param[out] dst    Output tensor info
     * @param[in]  window Region on which to execute the kernel.
     */
    using QuantizeDownFunctionPtr = void (CpuGemmLowpQuantizeDownInt32ToInt8ScaleByFixedPointKernel::*)(
                                        const ITensor *src, const ITensor *bias, ITensor *dst, const Window &window);

    QuantizeDownFunctionPtr _func{ nullptr };
    int                     _result_fixedpoint_multiplier{ 0 };
    int                     _result_shift{ 0 };
    int                     _result_offset_after_shift{ 0 };
    int                     _min{ 0 };
    int                     _max{ 0 };
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_GEMMLOWP_QUANTIZEDOWN_INT32TOINT8_SCALEBYFIXEDPOINT_KERNEL_H */
