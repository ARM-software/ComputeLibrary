/*
 * Copyright (c) 2021-2024 Arm Limited.
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
#ifndef ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUADD_H
#define ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUADD_H

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/runtime/NEON/INEOperator.h"

#include <memory>

namespace arm_compute
{
namespace experimental
{
namespace op
{
/** Wrapper class for CpuAdd. For information on the functions,
 * see "src/cpu/operators/CpuAdd.h"
*/
class CpuAdd : public INEOperator
{
public:
    /** Constructor */
    CpuAdd();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuAdd(const CpuAdd &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuAdd &operator=(const CpuAdd &) = delete;
    /** Default move constructor */
    CpuAdd(CpuAdd &&) = default;
    /** Default move assignment */
    CpuAdd &operator=(CpuAdd &&) = default;
    /** Default destructor */
    ~CpuAdd() override;
    /** Initialise the kernel's input, dst and border mode.
     *
     * Valid configurations (src0,src1) -> dst :
     *
     *   - (U8,U8)           -> U8
     *   - (S16,S16)         -> S16
     *   - (S32,S32)         -> S32
     *   - (F16,F16)         -> F16
     *   - (F32,F32)         -> F32
     *   - (QASYMM8,QASYMM8) -> QASYMM8
     *   - (QASYMM8_SIGNED,QASYMM8_SIGNED) -> QASYMM8_SIGNED
     *   - (QSYMM16,QSYMM16) -> QSYMM16
     *
     * @param[in]  src0     First input tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32
     * @param[in]  src1     Second input tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32
     * @param[out] dst      The dst tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/S32/F32.
     * @param[in]  policy   Overflow policy.
     * @param[in]  act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
     *
     */
    void configure(const ITensorInfo         *src0,
                   const ITensorInfo         *src1,
                   ITensorInfo               *dst,
                   ConvertPolicy              policy,
                   const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuAdd::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo         *src0,
                           const ITensorInfo         *src1,
                           const ITensorInfo         *dst,
                           ConvertPolicy              policy,
                           const ActivationLayerInfo &act_info = ActivationLayerInfo());

    void run(ITensorPack &tensors) override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace op
} // namespace experimental
} // namespace arm_compute
#endif // ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUADD_H
