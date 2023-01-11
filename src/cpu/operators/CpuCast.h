/*
 * Copyright (c) 2021, 2023 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_CAST_H
#define ARM_COMPUTE_CPU_CAST_H

#include "src/cpu/ICpuOperator.h"

namespace arm_compute
{
namespace cpu
{
/** Basic function to run @ref kernels::CpuCastKernel */
class CpuCast : public ICpuOperator
{
public:
    /** Configure operator for a given list of arguments
     *
     * Input data type must be different than output data type.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst                                             |
     * |:--------------|:-----------------------------------------------|
     * |QASYMM8_SIGNED | S16, S32, F32, F16                             |
     * |QASYMM8        | U16, S16, S32, F32, F16                        |
     * |U8             | U16, S16, S32, F32, F16                        |
     * |U16            | U8, U32                                        |
     * |S16            | QASYMM8_SIGNED, U8, S32                        |
     * |F16            | QASYMM8_SIGNED, QASYMM8, F32, S32, U8          |
     * |S32            | QASYMM8_SIGNED, QASYMM8, F16, F32, U8          |
     * |F32            | QASYMM8_SIGNED, QASYMM8, BFLOAT16, F16, S32, U8|
     *
     * @param[in]  src    The source tensor to convert. Data types supported: U8/S8/U16/S16/U32/S32/F16/F32.
     * @param[out] dst    The destination tensor. Data types supported: U8/S8/U16/S16/U32/S32/F16/F32.
     * @param[in]  policy Conversion policy.
     *
     * @deprecated Support for BFLOAT16 will be removed in 23.05 release
     *
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst, ConvertPolicy policy);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuCast::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, ConvertPolicy policy);
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_ACTIVATION_H */
