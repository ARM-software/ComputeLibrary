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
#ifndef SRC_CPU_OPERATORS_CPUADDMULADD
#define SRC_CPU_OPERATORS_CPUADDMULADD

#include "arm_compute/core/TensorInfo.h"

#include "src/cpu/ICpuOperator.h"
#include "src/cpu/operators/CpuDequantize.h"

namespace arm_compute
{
namespace cpu
{
/** Basic function to run @ref kernels::CpuAddMulAddKernel */
class CpuAddMulAdd : public ICpuOperator
{
public:
    /** Initialize the operator's inputs and outputs.
     *
     * Similar to @ref NEAddMulAdd::configure()
     *
     */
    void configure(const ITensorInfo *input1, const ITensorInfo *input2,
                   const ITensorInfo *bn_mul, const ITensorInfo *bn_add,
                   ITensorInfo *add_output, ITensorInfo *final_output,
                   ConvertPolicy policy, const ActivationLayerInfo &act_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuAddMulAdd::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2,
                           const ITensorInfo *bn_mul, const ITensorInfo *bn_add,
                           const ITensorInfo *add_output, const ITensorInfo *final_output,
                           ConvertPolicy policy, const ActivationLayerInfo &act_info);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;

    // We need auxilary memory to dequantize batchnorm coefficients
    experimental::MemoryRequirements workspace() const override;

private:
    enum AuxTensorIdx
    {
        DequantizedBnMul = 0,
        DequantizedBnAdd,
        Count
    };

    CpuDequantize _dequantize_bn_mul{};
    CpuDequantize _dequantize_bn_add{};

    TensorInfo _dequantized_bn_mul{};
    TensorInfo _dequantized_bn_add{};

    experimental::MemoryRequirements _aux_mem{ Count };
};
} // namespace cpu
} // namespace arm_compute
#endif /* SRC_CPU_OPERATORS_CPUADDMULADD */
