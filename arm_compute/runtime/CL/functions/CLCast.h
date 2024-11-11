/*
 * Copyright (c) 2018-2021, 2023-2024 Arm Limited.
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

#ifndef ACL_ARM_COMPUTE_RUNTIME_CL_FUNCTIONS_CLCAST_H
#define ACL_ARM_COMPUTE_RUNTIME_CL_FUNCTIONS_CLCAST_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to run type cast operation */
class CLCast : public IFunction
{
public:
    /** Constructor */
    CLCast();
    /** Destructor */
    ~CLCast();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLCast(const CLCast &) = delete;
    /** Default move constructor */
    CLCast(CLCast &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLCast &operator=(const CLCast &) = delete;
    /** Default move assignment operator */
    CLCast &operator=(CLCast &&);
    /** Initialize the function's source, destination
     *
     * @note When casting from/to quantized types the scale and zeroPoint are ignored
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src                |dst                                                                                             |
     * |:------------------|:-----------------------------------------------------------------------------------------------|
     * |U8                 | S8, U16, S16, U32, S32, F16, F32, QASYMM8_SIGNED, QSYMM8, QSYMM8_PER_CHANNEL, 16-bit Quantized |
     * |S8                 | U8, U16, S16, U32, S32, F16, F32, QASYMM8, 16-bit Quantized                                    |
     * |U16                | U8, S8, S16, U32, S32, F16, F32, 8-bit Quantized, QSYMM16                                      |
     * |S16                | U8, S8, U16, U32, S32, F16, F32, 8-bit Quantized, QASYMM16                                     |
     * |U32                | U8, S8, U16, S16, S32, F16, F32, All Quantized                                                 |
     * |S32                | U8, S8, U16, S16, U32, F16, F32, All Quantized                                                 |
     * |U64                | U8, S8, U16, S16, U32, S32, F16, F32, All Quantized                                            |
     * |S64                | U8, S8, U16, S16, U32, S32, F16, F32, All Quantized                                            |
     * |F16                | U8, S8, U16, S16, S32, U32, F32, All Quantized                                                 |
     * |F32                | U8, S8, U16, S16, S32, U32, F16, All Quantized                                                 |
     * |QASYMM8            | S8, U16, S16, U32, S32, F16, F32, QASYMM8_SIGNED, QSYMM8, QSYMM8_PER_CHANNEL, 16-bit Quantized |
     * |QASYMM8_SIGNED     | U8, U16, S16, U32, S32, F16, F32, QASYMM8, 16-bit Quantized                                    |
     * |QSYMM8             | U8, U16, S16, U32, S32, F16, F32, QASYMM8, 16-bit Quantized                                    |
     * |QSYMM8_PER_CHANNEL | U8, U16, S16, U32, S32, F16, F32, 16-bit Quantized                                             |
     * |QASYMM16           | U8, S8, U16, U32, S32, F16, F32, 8-bit Quantized, QSYMM16                                      |
     * |QSYMM16            | U8, S8, U16, U32, S32, F16, F32, 8-bit Quantized, QASYMM16                                     |
     *
     * Input data type must be different than output data type.
     *
     * @param[in]  input  The input tensor to convert.
     * @param[out] output The output tensor.
     * @param[in]  policy Conversion policy.
     */
    void configure(const ICLTensor *input, ICLTensor *output, ConvertPolicy policy);
    // Initialize the function's source, destination
    void
    configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, ConvertPolicy policy);
    /** Static function to check if given info will lead to a valid configuration of @ref CLCast
     *
     * Similar to @ref CLCast::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, ConvertPolicy policy);
    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif // ACL_ARM_COMPUTE_RUNTIME_CL_FUNCTIONS_CLCAST_H
