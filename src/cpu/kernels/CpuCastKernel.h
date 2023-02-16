/*
 * Copyright (c) 2016-2023 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_CAST_KERNEL_H
#define ARM_COMPUTE_CPU_CAST_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Casts a given tensor to a new type
 *
 * @note When casting between quantized types the scale and zeroPoint are ignored
 */
class CpuCastKernel : public ICpuKernel<CpuCastKernel>
{
private:
    using CastKernelPtr = std::add_pointer<void(const ITensor *, ITensor *, const ThreadInfo &, ConvertPolicy, const Window &)>::type;

public:
    CpuCastKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuCastKernel);
    /** Set the src and dst of the kernel
     *
     * Valid conversions src -> dst :
     *
     *   - QASYMM8_SIGNED -> S16, S32, F32, F16
     *   - QASYMM8        -> U16, S16, S32, F32, F16
     *   - U8             -> U16, S16, S32, F32, F16
     *   - U16            -> U8, U32
     *   - S16            -> QASYMM8_SIGNED, U8, S32
     *   - BFLOAT16       -> F32
     *   - F16            -> QASYMM8_SIGNED, QASYMM8, F32, S32, U8
     *   - S32            -> QASYMM8_SIGNED, QASYMM8, F16, F32, U8
     *   - F32            -> QASYMM8_SIGNED, QASYMM8, BFLOAT16, F16, S32, U8
     *
     * @param[in]  src    The src tensor to convert. Data types supported: QASYMM8_SIGNED/QASYMM8/U8/U16/S16/BFLOAT16/F16/F32.
     * @param[out] dst    The dst tensor. Data types supported: QASYMM8_SIGNED/QASYMM8/U8/U16/S16/U32/S32/BFLOAT16/F16/F32.
     * @param[in]  policy Conversion policy.
     *
     * @deprecated Support for BFLOAT16 will be removed in 23.05 release
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst, ConvertPolicy policy);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuCastKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, ConvertPolicy policy);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

    struct CastKernel
    {
        const char                          *name;
        const CastDataTypeISASelectorDataPtr is_selected;
        CastKernelPtr                        ukernel;
    };

    static const std::vector<CastKernel> &get_available_kernels();

private:
    ConvertPolicy _policy{ ConvertPolicy::SATURATE };
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_CAST_KERNEL_H */
