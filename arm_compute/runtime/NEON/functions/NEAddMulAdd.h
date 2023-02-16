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
#ifndef ARM_COMPUTE_RUNTIME_NEON_FUNCTIONS_NEADDMULADD
#define ARM_COMPUTE_RUNTIME_NEON_FUNCTIONS_NEADDMULADD

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <memory>

namespace arm_compute
{
class ITensor;
class ITensorInfo;

/** Function to compute Add+Mul+Add fused operation */
class NEAddMulAdd : public IFunction
{
public:
    /** Constructor */
    NEAddMulAdd(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEAddMulAdd(const NEAddMulAdd &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEAddMulAdd(NEAddMulAdd &&) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEAddMulAdd &operator=(const NEAddMulAdd &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEAddMulAdd &operator=(NEAddMulAdd &&) = delete;
    /** Destructor */
    ~NEAddMulAdd();
    /** Initialize the function's inputs and outputs.
     *
     * Valid data layouts:
     * - Any
     *
     * Valid data type configurations:
     * |input1         |input2         |bn_mul         |bn_add         |add_output     |final_output   |
     * |:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
     * |QASYMM8        |QASYMM8        |QASYMM8        |QASYMM8        |QASYMM8        |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |QASYMM8_SIGNED |QASYMM8_SIGNED |QASYMM8_SIGNED |QASYMM8_SIGNED |
     * |F16            |F16            |F16            |F16            |F16            |F16            |
     * |F32            |F32            |F32            |F32            |F32            |F32            |
     *
     * This is what this composite function (tailored for add followed by a batch norm operation) does:
     *      add_output <- input1 + input2 (add)
     *      final_output <- add_output * bn_mul + bn_add  (batch norm = mul+add)
     *
     * @param[in]  input1       First tensor input. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  input2       Second tensor input. Data types supported: Same as @p input.
     * @param[in]  bn_mul       The multiplication coefficient on the feature dimension. Data types supported: Same as @p input.
     *                          It's one dimensional tensor with size equal to the feature maps [FM]
     * @param[in]  bn_add       The addition coefficient on the feature dimension. Data types supported: Same as @p input.
     *                          It's one dimensional tensor with size equal to the feature maps [FM]
     * @param[out] add_output   Output of the first add. Data type supported: Same as @p input.
     * @param[out] final_output Output of the add+mul+add+act composite operation. Data type supported: Same as @p input.
     * @param[in]  policy       Policy to handle overflow
     * @param[in]  act_info     (Optional) Activation layer information in case of a fused activation.
     *
     */
    void configure(ITensor *input1, ITensor *input2, ITensor *bn_mul, ITensor *bn_add,
                   ITensor *add_output, ITensor *final_output,
                   ConvertPolicy policy, const ActivationLayerInfo &act_info);
    /** Static function to check if given info will lead to a valid configuration of @ref NEAddMulAdd
     *
     * Similar to @ref NEAddMulAdd::configure() except the arguments are @ref ITensorInfo * instead of @ref ITensor *
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2,
                           const ITensorInfo *bn_mul, const ITensorInfo *bn_add,
                           const ITensorInfo *add_output, const ITensorInfo *final_output,
                           ConvertPolicy policy, const ActivationLayerInfo &act_info);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_RUNTIME_NEON_FUNCTIONS_NEADDMULADD */
