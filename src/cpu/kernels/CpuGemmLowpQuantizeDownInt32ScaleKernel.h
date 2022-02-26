/*
 * Copyright (c) 2020-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_GEMMLOWP_QUANTIZEDOWN_INT32_SCALE_KERNEL_H
#define ARM_COMPUTE_CPU_GEMMLOWP_QUANTIZEDOWN_INT32_SCALE_KERNEL_H

#include "arm_compute/core/KernelDescriptors.h"
#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
// Forward declarations
class ITensor;
namespace cpu
{
namespace kernels
{
/** Kernel used to quantize down the int32 accumulator values of GEMMLowp to QASYMM8/QASYMM8_SIGNED
 *
 * This kernel takes a final int32 accumulator value (the output of @ref CpuGemmLowpMatrixMultiplyKernel), and processes it to obtain the final QASYMM8/QASYMM8_SIGNED value.
 * The following computations will be performed by the kernel:
 *
 *  -# Add offset terms to final result
 *  -# Multiply each entry of result by result_mult_int
 *  -# Add bias to final result if bias tensor is not a nullptr
 *  -# Shift the int32 accumulator by result_shift
 *  -# Clamp the value between the specified min and max bounds
 *  -# Clamp the resulting int32 values:
 *  -#  -to the [0..255] range and cast to QASYMM8.
 *  -#  -to the [-128..127] range and cast to QASYMM8_SIGNED.
 *
 */
class CpuGemmLowpQuantizeDownInt32ScaleKernel : public ICpuKernel<CpuGemmLowpQuantizeDownInt32ScaleKernel>
{
public:
    CpuGemmLowpQuantizeDownInt32ScaleKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuGemmLowpQuantizeDownInt32ScaleKernel);
    /** Initialise the kernel's input and output.
     *
     * @param[in]  src          Input tensor info. Data type supported: S32
     * @param[in]  bias         Biases tensor info. Only shared biases supported and it can be a nullptr if the biases addition is not required.
     *                          Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] dst          Output tensor info. Data type supported: Data type supported: QASYMM8/QASYMM8_SIGNED
     * @param[out] output_stage GEMMLowp output stage metadata.
     */
    void configure(ITensorInfo *src, ITensorInfo *bias, ITensorInfo *dst, const GEMMLowpOutputStageInfo *output_stage);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuGemmLowpQuantizeDownInt32ScaleKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *bias, const ITensorInfo *dst, const GEMMLowpOutputStageInfo *output_stage);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

private:
    /** Template function to run the NEGEMMLowpQuantizeDownInt32ScaleKernel
     *
     * @param[in]  src    Input tensor info
     * @param[in]  bias   Biases tensor info
     * @param[out] dst    Output tensor info
     * @param[in]  window Region on which to execute the kernel. (Must be a valid region of the window returned by window())
     */
    template <typename T>
    void run_internal(const ITensor *src, const ITensor *bias, ITensor *dst, const Window &window);

    /** Common signature for all the specialised CpuGemmLowpQuantizeDownInt32ScaleKernel functions
     *
     * @param[in]  src    Input tensor info
     * @param[in]  bias   Biases tensor info
     * @param[out] dst    Output tensor info
     * @param[in]  window Region on which to execute the kernel.
     */
    using QuantizeDownFunctionPtr = void (CpuGemmLowpQuantizeDownInt32ScaleKernel::*)(const ITensor *src, const ITensor *bias, ITensor *dst, const Window &window);

    QuantizeDownFunctionPtr        _func{ nullptr };
    const GEMMLowpOutputStageInfo *_output_stage{ nullptr };
    bool                           _is_bounded_relu{ false };
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_GEMMLOWP_QUANTIZEDOWN_INT32_SCALE_KERNEL_H */
