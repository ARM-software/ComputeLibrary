/*
 * Copyright (c) 2019-2023 Arm Limited.
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
#ifndef ARM_COMPUTE_NECAST_H
#define ARM_COMPUTE_NECAST_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/Types.h"

#include <memory>

namespace arm_compute
{
class ITensor;
class ITensorInfo;

/** Basic function to run @ref cpu::kernels::CpuCastKernel.
 * This function ignores the scale and zeroPoint of quanized tensors,so QASYMM8 input is treated as uint8 values.
 */
class NECast : public IFunction
{
public:
    /** Constructor */
    NECast();
    /** Destructor */
    ~NECast();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NECast(const NECast &) = delete;
    /** Default move constructor */
    NECast(NECast &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NECast &operator=(const NECast &) = delete;
    /** Default move assignment operator */
    NECast &operator=(NECast &&);
    /** Initialize the function's source, destination
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
     * Input data type must be different than output data type.
     *
     * @param[in]  input  The input tensor to convert. Data types supported: QASYMM8_SIGNED/QASYMM8/U8/U16/S16/F16/S32/F32.
     * @param[out] output The output tensor. Data types supported: QASYMM8_SIGNED/QASYMM8/U8/S8/U16/S16/U32/S32/BFLOAT16/F16/F32.
     * @param[in]  policy Conversion policy.
     */
    void configure(ITensor *input, ITensor *output, ConvertPolicy policy);
    /** Static function to check if given info will lead to a valid configuration of @ref NECast
     *
     * @param[in] input  Source tensor info. Data types supported: QASYMM8_SIGNED/QASYMM8/U8/U16/S16/F16/S32/F32.
     * @param[in] output Destination tensor info. Data type supported: QASYMM8_SIGNED/QASYMM8/U8/S8/U16/S16/U32/S32/BFLOAT16/F16/F32.
     * @param[in] policy Conversion policy.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, ConvertPolicy policy);

    // Inherited methods overridden
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NECAST_H*/
