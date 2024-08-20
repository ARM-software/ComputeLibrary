/*
 * Copyright (c) 2021, 2024 Arm Limited.
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

#ifndef ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUELEMENTWISE_H
#define ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUELEMENTWISE_H

#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/INEOperator.h"

#include <memory>

namespace arm_compute
{
namespace experimental
{
namespace op
{

/** Wrapper class for CpuElementwiseDivision. For information on the functions,
 * see "src/cpu/operators/CpuElementwise.h"
*/
class CpuElementwiseDivision : public INEOperator
{
public:
    /** Constructor */
    CpuElementwiseDivision();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuElementwiseDivision(CpuElementwiseDivision &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuElementwiseDivision &operator=(CpuElementwiseDivision &) = delete;
    /** Default move constructor */
    CpuElementwiseDivision(CpuElementwiseDivision &&) = default;
    /** Default move assignment */
    CpuElementwiseDivision &operator=(CpuElementwiseDivision &&) = default;
    /** Defualt destructor */
    ~CpuElementwiseDivision() override;
    /** Initialise the kernel's inputs, dst and conversion policy.
     *
     * @param[in, out] src0 First tensor input info. Data types supported: S32/F16/F32.
     * @param[in, out] src1 Second tensor input info. Data types supported: Same as @p src0.
     * @param[out]     dst  Output tensor info. Data types supported: Same as @p src0.
     */
    void configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuElementwiseDivision::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst);

    void run(ITensorPack &tensors) override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Wrapper class for CpuElementwiseMax. For information on the functions,
 * see "src/cpu/operators/CpuElementwise.h"
*/
class CpuElementwiseMax : public INEOperator
{
public:
    /** Constructor */
    CpuElementwiseMax();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuElementwiseMax(CpuElementwiseMax &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuElementwiseMax &operator=(CpuElementwiseMax &) = delete;
    /** Default move constructor */
    CpuElementwiseMax(CpuElementwiseMax &&) = default;
    /** Default move assignment */
    CpuElementwiseMax &operator=(CpuElementwiseMax &&) = default;
    /** Defualt destructor */
    ~CpuElementwiseMax() override;
    /** Configure the operator
     *
     * @param[in]  src0 The first source tensor information.
     * @param[in]  src1 The second source tensor information. With PRelu, this is used as alpha tensor.
     * @param[out] dst  The output tensor information.
     */
    void configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuElementwiseMax::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst);

    void run(ITensorPack &tensors) override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Wrapper class for CpuElementwiseMin. For information on the functions,
 * see "src/cpu/operators/CpuElementwise.h"
*/
class CpuElementwiseMin : public INEOperator
{
public:
    /** Constructor */
    CpuElementwiseMin();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuElementwiseMin(CpuElementwiseMin &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuElementwiseMin &operator=(CpuElementwiseMin &) = delete;
    /** Default move constructor */
    CpuElementwiseMin(CpuElementwiseMin &&) = default;
    /** Default move assignment */
    CpuElementwiseMin &operator=(CpuElementwiseMin &&) = default;
    /** Defualt destructor */
    ~CpuElementwiseMin() override;
    /** Configure the operator
     *
     * @param[in]  src0 The first source tensor information.
     * @param[in]  src1 The second source tensor information. With PRelu, this is used as alpha tensor.
     * @param[out] dst  The output tensor information.
     */
    void configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuElementwiseMin::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst);

    void run(ITensorPack &tensors) override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace op
} // namespace experimental
} // namespace arm_compute
#endif // ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUELEMENTWISE_H
