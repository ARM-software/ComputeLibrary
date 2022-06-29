/*
 * Copyright (c) 2021-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_WINOGRAD_CONV2D_KERNEL_H
#define ARM_COMPUTE_CPU_WINOGRAD_CONV2D_KERNEL_H

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/FunctionDescriptors.h"
#include "src/core/common/Macros.h"
#include "src/cpu/ICpuOperator.h"
#include "src/cpu/kernels/CpuWinogradConv2dKernel.h"
#include "src/cpu/kernels/assembly/gemm_common.hpp"
#include "src/cpu/operators/CpuActivation.h"
#include "src/cpu/operators/CpuGemm.h"
#include "src/cpu/operators/CpuPermute.h"
#include "src/cpu/operators/internal/CpuGemmAssemblyDispatch.h"

namespace arm_compute
{
namespace cpu
{
class CpuWinogradConv2d : public ICpuOperator
{
public:
    /** Constructor */
    CpuWinogradConv2d();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuWinogradConv2d);
    /** Destructor */
    ~CpuWinogradConv2d();

    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src0           |src1           |src2   |dst            |
     * |:--------------|:--------------|:------|:--------------|
     * |F16            |F16            |F16    |F16            |
     * |F32            |F32            |F32    |F32            |
     *
     * @param[in]  src              Source tensor Info. 3 lower dimensions represent a single input [width, height, IFM],
     *                              while every optional dimension from 4 and above represent a batch of inputs.
     *                              Data types supported: F16/F32.
     * @param[in]  weights          Weights tensor Info. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: Same as @p input.
     *                              Currently only 3x3 and 5x5 kernels are supported.
     * @param[in]  biases           Biases tensor Info. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p weights.
     * @param[out] dst              Destination tensor Info. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                              Data types supported: Same as @p input.
     * @param[in]  conv_info        Contains padding and stride information described in @ref PadStrideInfo. Currently only unit strides are supported.
     * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in]  enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                              available which may introduce a drop of accuracy as well. Default is false
     */
    void configure(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst, const PadStrideInfo &conv_info,
                   const ActivationLayerInfo &act_info         = ActivationLayerInfo(),
                   bool                       enable_fast_math = false);
    /** Static function to check if given info will lead to a valid configuration of @ref CpuWinogradConv2d
     *
     * Similar to CpuWinogradConv2d::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const PadStrideInfo &conv_info,
                           const ActivationLayerInfo &act_info         = ActivationLayerInfo(),
                           bool                       enable_fast_math = false);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;
    void prepare(ITensorPack &constants) override;
    experimental::MemoryRequirements workspace() const override;

private:
    enum AuxTensorIdx
    {
        GemmWorkspace      = 0,
        Pretranspose       = 1,
        InterleavedLHS     = 2,
        TransposedRHS      = 3,
        TempResult         = 4,
        TransformedInput   = 5,
        TransformedOutput  = 6,
        WorkspaceIO        = 7,
        TransformedWeights = 8,
        PermutedWeights    = 9,
        PermutedInput      = TransformedOutput,
        PermutedOutput     = TransformedInput,
        Count              = 10
    };
    std::unique_ptr<CpuGemm>                   _gemm_function;
    std::unique_ptr<CpuActivation>             _activation_func;
    std::unique_ptr<ICPPKernel>                _transform_input_kernel;
    std::unique_ptr<ICPPKernel>                _transform_output_kernel;
    std::unique_ptr<CpuPermute>                _permute_input;
    std::unique_ptr<CpuPermute>                _permute_output;
    std::unique_ptr<CpuPermute>                _permute_weights;
    experimental::MemoryRequirements           _aux_mem{ Count };
    std::unique_ptr<arm_conv::ConvolutionArgs> _conv_args; // Make it unique ptr because this type does not have a default constructor
    arm_conv::winograd::WinogradImpl           _winograd_impl;
    DataLayout                                 _data_layout;
    TensorInfo                                 _winograd_transformed_input;
    TensorInfo                                 _winograd_transformed_output;
    TensorInfo                                 _winograd_transformed_weights;
    TensorInfo                                 _input_workspace;
    TensorInfo                                 _output_workspace;
    TensorInfo                                 _weights_hwio;
    TensorInfo                                 _input_nhwc;
    TensorInfo                                 _output_nhwc;
    bool                                       _is_prepared;
    bool                                       _run_activation;
};
} // namespace cpu
} // namespace arm_compute

#endif /* ARM_COMPUTE_CPU_WINOGRAD_CONV2D_KERNEL_H */
