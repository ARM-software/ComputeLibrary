/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_PIXELWISE_MULTIPLICATION_KERNEL_H
#define ARM_COMPUTE_CPU_PIXELWISE_MULTIPLICATION_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/core/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Interface for the kernel to perform addition between two tensors */
class CpuPixelWiseMultiplicationKernel : public ICpuKernel
{
public:
    /** Default constructor */
    CpuPixelWiseMultiplicationKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuPixelWiseMultiplicationKernel);
    /** Initialise the kernel's input, dst and border mode.
     *
     * Valid configurations (Src1,Src2) -> Dst :
     *
     *                                                       Support: Broadcast? Scale=1/255?
     *   - (U8,U8)                         -> U8, S16                 N          Y
     *   - (U8,S16)                        -> S16                     N          Y
     *   - (S16,U8)                        -> S16                     N          Y
     *   - (S16,S16)                       -> S16                     N          Y
     *   - (S32,S32)                       -> S32                     Y          N
     *   - (F16,F16)                       -> F16                     N          Y
     *   - (F32,F32)                       -> F32                     Y          Y
     *   - (QASYMM8,QASYMM8)               -> QASYMM8                 Y          Y
     *   - (QASYMM8_SIGNED,QASYMM8_SIGNED) -> QASYMM8_SIGNED          Y          Y
     *   - (QSYMM16,QSYMM16)               -> QSYMM16, S32            N          Y
     *
     * @note For @p scale equal to 1/255 only round to nearest even (implemented as round half up) is supported.
     *       For all other scale values only round to zero (implemented as round towards minus infinity) is supported.
     *
     * @param[in]  src1            First input tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/QSYMM16/F16/F32
     * @param[in]  src2            Second input tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/QSYMM16/F16/F32
     * @param[out] dst             Dst tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/QSYMM16/F16/F32
     * @param[in]  scale           Scale to apply after multiplication.
     *                             Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15.
     *                             If both @p src1, @p src2 and @p dst are of datatype S32, scale cannot be 1/255
     * @param[in]  overflow_policy Overflow policy. ConvertPolicy cannot be WRAP if any of the inputs is of quantized datatype
     * @param[in]  rounding_policy Rounding policy.
     */
    void configure(ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst, float scale, ConvertPolicy overflow_policy, RoundingPolicy rounding_policy);
    /** Static function to check if given info will lead to a valid configuration of @ref CpuPixelWiseMultiplicationKernel
     *
     * Valid configurations (Src1,Src2) -> Dst :
     *                                                       Support: Broadcast? Scale=1/255?
     *   - (U8,U8)                         -> U8, S16                 N          Y
     *   - (U8,S16)                        -> S16                     N          Y
     *   - (S16,U8)                        -> S16                     N          Y
     *   - (S16,S16)                       -> S16                     N          Y
     *   - (S32,S32)                       -> S32                     Y          N
     *   - (F16,F16)                       -> F16                     N          Y
     *   - (F32,F32)                       -> F32                     Y          Y
     *   - (QASYMM8,QASYMM8)               -> QASYMM8                 Y          Y
     *   - (QASYMM8_SIGNED,QASYMM8_SIGNED) -> QASYMM8_SIGNED          Y          Y
     *   - (QSYMM16,QSYMM16)               -> QSYMM16, S32            N          Y
     *
     * @note For @p scale equal to 1/255 only round to nearest even (implemented as round half up) is supported.
     *       For all other scale values only round to zero (implemented as round towards minus infinity) is supported.
     *
     * @param[in] src1            First src tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/QSYMM16/F16/F32
     * @param[in] src2            Second src tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/QSYMM16/F16/F32
     * @param[in] dst             Dst tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/QSYMM16/F16/F32
     * @param[in] scale           Scale to apply after multiplication.
     *                            Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15.
     *                            If both @p src1, @p src2 and @p dst are of datatype S32, scale cannot be 1/255
     * @param[in] overflow_policy Overflow policy. ConvertPolicy cannot be WRAP if any of the srcs is of quantized datatype
     * @param[in] rounding_policy Rounding policy.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, float scale, ConvertPolicy overflow_policy, RoundingPolicy rounding_policy);

    // Inherited methods overridden
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

private:
    /** Common signature for all the specialised multiplication functions with integer scaling factor
     *
     * @param[in]  src1   Src1 tensor object.
     * @param[in]  src2   Src2 tensor object.
     * @param[out] dst    Dst tensor object.
     * @param[in]  window Region on which to execute the kernel
     * @param[in]  scale  Integer scale factor.
     */
    using MulFunctionInt = void(const ITensor *src1, const ITensor *src2, ITensor *dst, const Window &window, int scale);
    /** Common signature for all the specialised multiplication functions with float scaling factor
     *
     * @param[in]  src1   Src1 tensor object.
     * @param[in]  src2   Src2 tensor object.
     * @param[out] dst    Dst tensor object.
     * @param[in]  window Region on which to execute the kernel
     * @param[in]  scale  Float scale factor.
     */
    using MulFunctionFloat = void(const ITensor *src1, const ITensor *src2, ITensor *dst, const Window &window, float scale);
    /** Common signature for all the specialised QASYMM8 multiplication functions with float scaling factor
     *
     * @param[in]  src1   Src1 tensor object.
     * @param[in]  src2   Src2 tensor object.
     * @param[out] dst    Dst tensor object.
     * @param[in]  window Region on which to execute the kernel
     * @param[in]  scale  Float scale factor.
     *
     */
    using MulFunctionQuantized = void(const ITensor *src1, const ITensor *src2, ITensor *dst, const Window &window, float scale);

    MulFunctionFloat     *_func_float{ nullptr };
    MulFunctionInt       *_func_int{ nullptr };
    MulFunctionQuantized *_func_quantized{ nullptr };
    float                 _scale{ 0 };
    int                   _scale_exponent{ 0 };
};

/** Interface for the complex pixelwise multiplication kernel. */
class CpuComplexPixelWiseMultiplicationKernel : public ICpuKernel
{
public:
    /** Default constructor */
    CpuComplexPixelWiseMultiplicationKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuComplexPixelWiseMultiplicationKernel);
    /** Initialise the kernel's src, dst and border mode.
     *
     * @param[in]  src1 An src tensor. Data types supported: F32. Number of channels supported: 2 (complex tensor).
     * @param[in]  src2 An src tensor. Data types supported: same as @p src1. Number of channels supported: same as @p src1.
     * @param[out] dst  The dst tensor, Data types supported: same as @p src1.  Number of channels supported: same as @p src1.
     */
    void configure(ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration of @ref CpuComplexPixelWiseMultiplicationKernel
     *
     * @param[in] src1 An src tensor info. Data types supported: F32. Number of channels supported: 2 (complex tensor).
     * @param[in] src2 An src tensor info. Data types supported: same as @p src1. Number of channels supported: same as @p src1.
     * @param[in] dst  The dst tensor info. Data types supported: same as @p src1. Number of channels supported: same as @p src1.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /*ARM_COMPUTE_CPU_PIXELWISE_MULTIPLICATION_KERNEL_H */
