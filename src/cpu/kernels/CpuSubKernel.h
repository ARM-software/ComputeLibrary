/*
 * Copyright (c) 2016-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_SUB_KERNEL_H
#define ARM_COMPUTE_CPU_SUB_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Interface for the kernel to perform subtraction between two tensors */
class CpuSubKernel : public ICpuKernel<CpuSubKernel>
{
private:
    using SubKernelPtr = std::add_pointer<void(const ITensor *, const ITensor *, ITensor *, const ConvertPolicy &, const Window &)>::type;

public:
    CpuSubKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuSubKernel);

    /** Initialise the kernel's src and dst.
     *
     * Valid configurations (src0,src1) -> dst :
     *
     *   - (U8,U8)                          -> U8
     *   - (QASYMM8, QASYMM8)               -> QASYMM8
     *   - (QASYMM8_SIGNED, QASYMM8_SIGNED) -> QASYMM8_SIGNED
     *   - (S16,S16)                        -> S16
     *   - (S32,S32)                        -> S32
     *   - (F16,F16)                        -> F16
     *   - (F32,F32)                        -> F32
     *
     * @param[in]  src0   An input tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/S32/F16/F32
     * @param[in]  src1   An input tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/S32/F16/F32
     * @param[out] dst    The dst tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/S32/F16/F32.
     * @param[in]  policy Overflow policy. Convert policy cannot be WRAP if datatype is quantized.
     */
    void configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst, ConvertPolicy policy);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuSubKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst, ConvertPolicy policy);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

    struct SubKernel
    {
        const char                  *name;
        const DataTypeISASelectorPtr is_selected;
        SubKernelPtr                 ukernel;
    };

    static const std::vector<SubKernel> &get_available_kernels();

private:
    ConvertPolicy _policy{};
    SubKernelPtr  _run_method{ nullptr };
    std::string   _name{};
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_SUB_KERNEL_H */
