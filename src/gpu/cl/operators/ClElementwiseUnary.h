/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CL_ELEMENTWISE_UNARY_H
#define ARM_COMPUTE_CL_ELEMENTWISE_UNARY_H

#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClOperator.h"

namespace arm_compute
{
namespace opencl
{
/** Basic function to perform inverse square root on an src tensor. */
class ClRsqrt : public IClOperator
{
public:
    /** Initialize the function
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             Source tensor info. Data types supported: F16/F32.
     * @param[out] dst             Destination tensor info. Data types supported: same as @p src.
     */
    void configure(const ClCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClRsqrt::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);
};

/** Basic function to perform exponential on an src tensor. */
class ClExp : public IClOperator
{
public:
    /** Initialize the function
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             Source tensor info. Data types supported: F16/F32.
     * @param[out] dst             Destination tensor info. Data types supported: same as @p src.
     */
    void configure(const ClCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClExp::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);
};

/** Basic function to negate an src tensor. */
class ClNeg : public IClOperator
{
public:
    /** Initialize the function
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             Source tensor info. Data types supported: F16/F32.
     * @param[out] dst             Destination tensor info. Data types supported: same as @p src.
     */
    void configure(const ClCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClNeg::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);
};

/** Basic function to calculate sine of an src tensor. */
class ClSin : public IClOperator
{
public:
    /** Initialize the function
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             Source tensor info. Data types supported: F16/F32.
     * @param[out] dst             Destination tensor info. Data types supported: same as @p src.
     */
    void configure(const ClCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClSin::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);
};

/** Basic function to perform elementwise log on an src tensor. */
class ClLog : public IClOperator
{
public:
    /** Initialize the function
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             Source tensor info. Data types supported: F16/F32.
     * @param[out] dst             Destination tensor info. Data types supported: same as @p src.
     */
    void configure(const ClCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClLog::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);
};

/** Basic function to get the absolute value of an src tensor. */
class ClAbs : public IClOperator
{
public:
    /** Initialize the function
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             Source tensor info. Data types supported: F16/F32.
     * @param[out] dst             Destination tensor info. Data types supported: same as @p src.
     */
    void configure(const ClCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClAbs::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);
};

/** Basic function to get the round (to the nearest even) value of an src tensor. */
class ClRound : public IClOperator
{
public:
    /** Initialize the function
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             Source tensor info. Data types supported: F16/F32.
     * @param[out] dst             Destination tensor info. Data types supported: same as @p src.
     */
    void configure(const ClCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClRound::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);
};
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_ELEMENTWISE_UNARY_H */
