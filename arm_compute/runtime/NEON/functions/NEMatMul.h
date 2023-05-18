/*
 * Copyright (c) 2023 Arm Limited.
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
#ifndef ACL_ARM_COMPUTE_RUNTIME_NEON_FUNCTIONS_NEMATMUL
#define ACL_ARM_COMPUTE_RUNTIME_NEON_FUNCTIONS_NEMATMUL

#include "arm_compute/runtime/IFunction.h"
#include <memory>

namespace arm_compute
{
/** Settings for MatMul Cpu implementation*/
class CpuMatMulSettings
{
public:
    // get fast math flag
    bool fast_math() const
    {
        return _fast_math;
    }
    // Set fast math flag
    CpuMatMulSettings &fast_math(bool fmath)
    {
        _fast_math = fmath;
        return *this;
    };

private:
    bool _fast_math{ false };
};

// Forward declarations
class ITensor;
class ITensorInfo;
class MatMulInfo;
class Status;

/** Basic function to run the following operators:
 *
 * -# @ref cpu::CpuMatMul
 */
class NEMatMul : public IFunction
{
public:
    /** Constructor */
    NEMatMul();
    /** Destructor */
    ~NEMatMul();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEMatMul(const NEMatMul &) = delete;
    /** Default move constructor */
    NEMatMul(NEMatMul &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEMatMul &operator=(const NEMatMul &) = delete;
    /** Default move assignment operator */
    NEMatMul &operator=(NEMatMul &&) = default;
    /** Initialize
     *
     * Valid data layouts:
     * - Any
     *
     * Valid data type configurations:
     * |lhs            |rhs                |dst            |
     * |:--------------|:------------------|:--------------|
     * |F32            |F32                |F32            |
     * |F16            |F16                |F16            |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED     |QASYMM8_SIGNED |
     * |QASYMM8        |QASYMM8            |QASYMM8        |
     *
     * @param[in]  lhs      Left-hand side tensor info. Data types supported: F16/F32/QASYMM8_SIGNED/QASYMM8.
     * @param[in]  rhs      Right-hand side tensor info. Data types supported: same as @p lhs.
     * @param[out] dst      Output tensor to store the result of the batched matrix multiplication. Data types supported: same as @p lhs / @p rhs.
     * @param[in]  info     Contains MatMul operation information described in @ref MatMulInfo.
     * @param[in]  settings Class containing flags for function level settings i.e fast math
     */
    void configure(ITensor *lhs, ITensor *rhs, ITensor *dst, const MatMulInfo &info, const CpuMatMulSettings &settings);
    /** Static function to check if given info will lead to a valid configuration of @ref NEMatMul
     *
     * Parameters are similar to @ref NEMatMul::configure()
     *
     * @return Status
     */
    static Status validate(const ITensorInfo *lhs, const ITensorInfo *rhs, const ITensorInfo *dst, const MatMulInfo &info, const CpuMatMulSettings &settings);

    // Inherited methods overridden
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
}
#endif /* ACL_ARM_COMPUTE_RUNTIME_NEON_FUNCTIONS_NEMATMUL */
