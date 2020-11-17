/*
 * Copyright (c) 2020 Arm Limited.
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
#ifndef ARM_COMPUTE_NELOGICAL_H
#define ARM_COMPUTE_NELOGICAL_H

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/Macros.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class ITensor;
class ITensorInfo;

/** Basic function to perform logical AND */
class NELogicalAnd : public IFunction
{
public:
    /** Constructor */
    NELogicalAnd();
    /** Destructor */
    ~NELogicalAnd();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE_INC(NELogicalAnd)

    /** Initialise the kernel's inputs and output
     *
     * @param[in]  input1 First tensor input. Data type supported: U8.
     * @param[in]  input2 Second tensor input. Data type supported: U8.
     * @param[out] output Output tensor. Data type supported: U8.
     */
    void configure(const ITensor *input1, const ITensor *input2, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NELogicalAnd
     *
     * @param[in] input1 First input tensor info. Data types supported: U8.
     * @param[in] input2 Second input tensor info. Data types supported: U8.
     * @param[in] output Output tensor info. Data type supported: U8
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);

    // Inherited methods overridden
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Basic function to perform logical OR */
class NELogicalOr : public IFunction
{
public:
    /** Constructor */
    NELogicalOr();
    /** Destructor */
    ~NELogicalOr();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE_INC(NELogicalOr)

    /** Initialise the kernel's inputs and output
     *
     * @param[in]  input1 First tensor input. Data type supported: U8.
     * @param[in]  input2 Second tensor input. Data type supported: U8.
     * @param[out] output Output tensor. Data type supported: U8.
     */
    void configure(const ITensor *input1, const ITensor *input2, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NELogicalOr
     *
     * @param[in] input1 First input tensor info. Data types supported: U8.
     * @param[in] input2 Second input tensor info. Data types supported: U8.
     * @param[in] output Output tensor info. Data type supported: U8
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);

    // Inherited methods overridden
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Basic function to perform logical NOT */
class NELogicalNot : public IFunction
{
public:
    /** Constructor */
    NELogicalNot();
    /** Destructor */
    ~NELogicalNot();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE_INC(NELogicalNot)

    /** Initialise the kernel's inputs and output
     *
     * @param[in]  input  Input tensor. Data type supported: U8.
     * @param[out] output Output tensor. Data type supported: U8.
     */
    void configure(const ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NELogicalNot
     *
     * @param[in] input  Input tensor info. Data types supported: U8.
     * @param[in] output Output tensor info. Data type supported: U8
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NELOGICAL_H */
