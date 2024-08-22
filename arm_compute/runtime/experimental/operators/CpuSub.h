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
#ifndef ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUSUB_H
#define ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUSUB_H

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/function_info/ActivationLayerInfo.h"
#include "arm_compute/runtime/NEON/INEOperator.h"

#include <memory>

namespace arm_compute
{
namespace experimental
{
namespace op
{

/** Wrapper class for CpuSub. For information on the functions,
 * see "src/cpu/operators/CpuSub.h"
*/
class CpuSub : public INEOperator
{
public:
    /** Constructor */
    CpuSub();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuSub(const CpuSub &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuSub &operator=(const CpuSub &) = delete;
    /** Default move constructor */
    CpuSub(CpuSub &&) = default;
    /** Default move assignment */
    CpuSub &operator=(CpuSub &&) = default;
    /** Default destructor */
    ~CpuSub() override;
    /** Initialise the kernel's inputs, dst and conversion policy.
     *
     * Valid configurations (src0,src1) -> dst :
     *
     *   - (U8,U8)                          -> U8
     *   - (QASYMM8, QASYMM8)               -> QASYMM8
     *   - (QASYMM8_SIGNED, QASYMM8_SIGNED) -> QASYMM8_SIGNED
     *   - (S16,S16)                        -> S16
     *   - (S32,S32)                        -> S32
     *   - (F16,F16)                        -> F16
     *   - (F32,F32)                        -> F32
     *
     * @param[in]  src0     First tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/S32/F16/F32
     * @param[in]  src1     Second tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/S32/F16/F32
     * @param[out] dst      Output tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/QSYMM16/S16/S32/F16/F32
     * @param[in]  policy   Policy to use to handle overflow. Convert policy cannot be WRAP if datatype is quantized.
     * @param[in]  act_info (Optional) Activation layer information in case of a fused activation. Currently not supported.
     */
    void configure(const ITensorInfo         *src0,
                   const ITensorInfo         *src1,
                   ITensorInfo               *dst,
                   ConvertPolicy              policy,
                   const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuSub::configure()
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
#endif // ACL_ARM_COMPUTE_RUNTIME_EXPERIMENTAL_OPERATORS_CPUSUB_H
