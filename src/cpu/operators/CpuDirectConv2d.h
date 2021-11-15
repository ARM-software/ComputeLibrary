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
#ifndef ARM_COMPUTE_CPU_DIRECTCONV2D_H
#define ARM_COMPUTE_CPU_DIRECTCONV2D_H

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "src/core/NEON/kernels/NEFillBorderKernel.h"
#include "src/cpu/ICpuKernel.h"
#include "src/cpu/ICpuOperator.h"
#include "src/cpu/kernels/CpuDirectConv2dKernel.h"
#include "src/cpu/kernels/CpuDirectConv2dOutputStageKernel.h"
#include "src/cpu/operators/CpuActivation.h"

#include <memory>

namespace arm_compute
{
namespace cpu
{
/** Function to run the direct convolution.
 *
 *  This function calls the following kernels:
 *
 * -# @ref NEFillBorderKernel for the input
 * -# @ref kernels::CpuDirectConv2dOutputStageKernel
 * -# @ref kernels::CpuDirectConv2dKernel
 */
class CpuDirectConv2d : public ICpuOperator
{
public:
    CpuDirectConv2d(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    ~CpuDirectConv2d();
    /** Set the input, weights, biases and output tensors.
     *
     * @note: DirectConvolution only works in the following configurations:
     *    1x1 convolution with stride_x = 1/2/3, stride_y = 1/2/3 data type = F16/F32
     *    3x3 convolution with stride_x = 1/2/3, stride_y = 1/2/3 data type = F16/F32
     *    5x5 convolution with stride_x = 1/2/3, stride_y = 1/2/3 data type = F32
     *
     * @param[in, out] src       Input tensor info. Data types supported: F16/F32.
     * @param[in]      weights   Set of kernels to convolve the input volume.
     *                           Supported sizes: 1x1, 3x3 and 5x5.
     *                           The 3rd dimension must be the same as the input's volume 3rd dimension.
     *                           Data type supported: Same as @p src.
     * @param[in]      bias      Set of biases. Can be nullptr. Data type supported: Same as @p src.
     * @param[out]     dst       Output tensor info.
     *                           The 3rd dimensions must be equal to the 4th dimension of the @p kernels tensor. Data types supported: Same as @p input.
     * @param[in]      conv_info Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]      act_info  (Optional) Activation layer information in case of a fused activation.
     */
    void configure(ITensorInfo *src, ITensorInfo *weights, const ITensorInfo *bias, ITensorInfo *dst, const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuDirectConv2d::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *bias, const ITensorInfo *dst, const PadStrideInfo &conv_info,
                           const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;

private:
    MemoryGroup                                                _memory_group;
    std::unique_ptr<kernels::CpuDirectConv2dOutputStageKernel> _output_stage_kernel;
    std::unique_ptr<kernels::CpuDirectConv2dKernel>            _conv_kernel;
    std::unique_ptr<NEFillBorderKernel>                        _input_border_handler;
    std::unique_ptr<CpuActivation>                             _activationlayer_function;
    Tensor                                                     _accumulator;
    bool                                                       _has_bias{ false };
    bool                                                       _is_activationlayer_enabled{ false };
    unsigned int                                               _dim_split{ 0 };
    bool                                                       _is_padding_required{ false };
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_DIRECTCONV2D_H */
