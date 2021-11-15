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
#ifndef ARM_COMPUTE_CPU_GEMM_DIRECT_CONV_2D_H
#define ARM_COMPUTE_CPU_GEMM_DIRECT_CONV_2D_H

#include "arm_compute/core/TensorInfo.h"
#include "src/core/common/Macros.h"
#include "src/cpu/ICpuOperator.h"
#include "src/cpu/operators/CpuActivation.h"
#include "src/cpu/operators/CpuPermute.h"
#include "src/cpu/operators/internal/CpuGemmAssemblyDispatch.h"

namespace arm_compute
{
// Forward declarations
class ITensor;
struct Conv2dInfo;
namespace cpu
{
class CpuGemmDirectConv2d : public ICpuOperator
{
public:
    CpuGemmDirectConv2d();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuGemmDirectConv2d);
    ~CpuGemmDirectConv2d();
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |src1           |src2           |dst            |
     * |:--------------|:--------------|:--------------|:--------------|
     * |QASYMM8        |QASYMM8        |S32            |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |S32            |QASYMM8_SIGNED |
     * |F16            |F16            |F16            |F16            |
     * |F32            |F32            |F32            |F32            |
     * |BFLOAT16       |BFLOAT16       |BFLOAT16       |BFLOAT16       |
     *
     * @param[in] src     Source tensor info. 3 lower dimensions represent a single input [width, height, IFM],
     *                    while every optional dimension from 4 and above represent a batch of inputs.
     *                    Data types supported: QASYMM8/QASYMM8_SIGNED/BFLOAT16/F16/F32.
     * @param[in] weights Weights tensor info. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                    Data type supported: QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL/BFLOAT16/F16/F32.
     * @param[in] biases  Biases tensor info. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                    Data type supported: Should match @p input data type, except for input of QASYMM8/QASYMM8_SIGNED type where biases should be of S32 type.
     * @param[in] dst     Destination tensor info. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                    Data types supported: Same as @p input.
     * @param[in] info    Contains padding and stride information described in @ref PadStrideInfo.
     */
    void configure(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst, const Conv2dInfo &info);
    /** Static function to check if given info will lead to a valid configuration of @ref CpuGemmDirectConv2d
     *
     * Similar to CpuGemmDirectConv2d::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const Conv2dInfo &info);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
    void prepare(ITensorPack &constants) override;
    experimental::MemoryRequirements workspace() const override;

private:
    enum AuxTensorIdx
    {
        AsmGemmWorkspace = 0,
        Pretranspose,
        PermutedWeights,
        Count
    };

    std::unique_ptr<CpuGemmAssemblyDispatch> _gemm_asm_func;
    std::unique_ptr<CpuActivation>           _activation_func;
    std::unique_ptr<CpuPermute>              _weights_permute_func;
    experimental::MemoryRequirements         _aux_mem;
    TensorInfo                               _perm_weights;
    bool                                     _run_activation;
    bool                                     _is_prepared;
};
} // namespace cpu
} // namespace arm_compute

#endif /* ARM_COMPUTE_CPU_GEMM_DIRECT_CONV_2D_H */
