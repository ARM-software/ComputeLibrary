/*
 * Copyright (c) 2017-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_QUANTIZE_KERNEL_H
#define ARM_COMPUTE_CPU_QUANTIZE_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Interface for the quantization layer kernel.
 *
 * @note The implementation supports only 3D input tensors
 */
class CpuQuantizeKernel : public ICpuKernel<CpuQuantizeKernel>
{
public:
    CpuQuantizeKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuQuantizeKernel);
    /** Set the input, output.
     *
     * @param[in]  src Source tensor info. The dimensions over the third will be interpreted as batches. Data types supported: QASYMM8/QASYMM8_SIGNED/F32/F16.
     * @param[out] dst Destination tensor info with the same dimensions of input. Data types supported: QASYMM8/QASYMM8_SIGNED/QASYMM16.
     *
     * @note Output auto initialization is not supported by this kernel
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuQuantizeKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

private:
    /** Common signature for all the specialised @ref CpuQuantizeKernel functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using QuantizeFunctionExecutorPtr = void (CpuQuantizeKernel::*)(const ITensor *src, ITensor *dst, const Window &window);
    /** Function to apply QASYMM8 or QASYMM8_SIGNED quantization on a tensor.
     *
     * @param[in] window Region on which to execute the kernel.
     */
    template <typename TIn, typename TOut>
    void run_quantize_qasymm8(const ITensor *src, ITensor *dst, const Window &window);
    /** Function to apply QASYMM16 quantization on a tensor.
     *
     * @param[in] window Region on which to execute the kernel.
     */
    template <typename T>
    void run_quantize_qasymm16(const ITensor *src, ITensor *dst, const Window &window);

    template <typename TIn, typename TOut>
    void run_quantize_qsymm8(const ITensor *src, ITensor *dst, const Window &window);

    QuantizeFunctionExecutorPtr _func{ nullptr };
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_QUANTIZE_KERNEL_H */
