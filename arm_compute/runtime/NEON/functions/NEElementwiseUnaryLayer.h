/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEELEMENTWISEUNARYLAYER_H__
#define __ARM_COMPUTE_NEELEMENTWISEUNARYLAYER_H__

#include "arm_compute/runtime/NEON/INESimpleFunction.h"

namespace arm_compute
{
class ITensor;

/** Basic function to perform inverse square root on an input tensor. */
class NERsqrtLayer : public INESimpleFunction
{
public:
    /** Initialize the function
     *
     * @param[in]  input  Input tensor. Data types supported: F16/F32.
     * @param[out] output Output tensor. Data types supported: same as @p input.
     */
    void configure(const ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NERsqrtLayer
     *
     * @param[in] input  First tensor input info. Data types supported: F16/F32.
     * @param[in] output Output tensor info. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);
};

/** Basic function to perform exponential on an input tensor. */
class NEExpLayer : public INESimpleFunction
{
public:
    /** Initialize the function
     *
     * @param[in]  input  Input tensor. Data types supported: F16/F32.
     * @param[out] output Output tensor. Data types supported: same as @p input.
     */
    void configure(const ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEExpLayer
     *
     * @param[in] input  First tensor input info. Data types supported: F16/F32.
     * @param[in] output Output tensor info. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);
};

/** Basic function to negate an input tensor. */
class NENegLayer : public INESimpleFunction
{
public:
    /** Initialize the function
     *
     * @param[in]  input  Input tensor. Data types supported: F16/F32/S32.
     * @param[out] output Output tensor. Data types supported: same as @p input.
     */
    void configure(const ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NENegLayer
     *
     * @param[in] input  First tensor input info. Data types supported: F16/F32/S32.
     * @param[in] output Output tensor info. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);
};

/** Basic function to compute the natural logarithm of an input tensor. */
class NELogLayer : public INESimpleFunction
{
public:
    /** Initialize the function
     *
     * @param[in]  input  Input tensor. Data types supported: F16/F32/S32.
     * @param[out] output Output tensor. Data types supported: same as @p input.
     */
    void configure(const ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NELogLayer
     *
     * @param[in] input  First tensor input info. Data types supported: F16/F32/S32.
     * @param[in] output Output tensor info. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);
};

/** Basic function to compute the absolute value of an input tensor. */
class NEAbsLayer : public INESimpleFunction
{
public:
    /** Initialize the function
     *
     * @param[in]  input  Input tensor. Data types supported: F16/F32/S32.
     * @param[out] output Output tensor. Data types supported: same as @p input.
     */
    void configure(const ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEAbsLayer
     *
     * @param[in] input  First tensor input info. Data types supported: F16/F32/S32.
     * @param[in] output Output tensor info. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);
};

/** Basic function to compute the round value elementwise of an input tensor. */
class NERoundLayer : public INESimpleFunction
{
public:
    /** Initialize the function
     *
     * @param[in]  input  Input tensor. Data types supported: F16/F32.
     * @param[out] output Output tensor. Data types supported: same as @p input.
     */
    void configure(const ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NERoundLayer
     *
     * @param[in] input  First tensor input info. Data types supported: F16/F32.
     * @param[in] output Output tensor info. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);
};

/** Basic function to compute the sine of an input tensor. */
class NESinLayer : public INESimpleFunction
{
public:
    /** Initialize the function
     *
     * @param[in]  input  Input tensor. Data types supported: F16/F32.
     * @param[out] output Output tensor. Data types supported: same as @p input.
     */
    void configure(const ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NESinLayer
     *
     * @param[in] input  First tensor input info. Data types supported: F16/F32.
     * @param[in] output Output tensor info. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEELEMENTWISEUNARYLAYER_H__ */
