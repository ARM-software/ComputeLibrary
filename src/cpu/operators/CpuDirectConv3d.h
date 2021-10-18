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
#ifndef ARM_COMPUTE_CPU_DIRECTCONV3D_H
#define ARM_COMPUTE_CPU_DIRECTCONV3D_H

#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/runtime/FunctionDescriptors.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "src/core/NEON/kernels/NEFillBorderKernel.h"
#include "src/cpu/ICpuKernel.h"
#include "src/cpu/ICpuOperator.h"
#include "src/cpu/kernels/CpuDirectConv3dKernel.h"
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
 * -# @ref kernels::CpuDirectConv3dKernel
 */
class CpuDirectConv3d : public ICpuOperator
{
public:
    CpuDirectConv3d(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    ~CpuDirectConv3d();
    /** Set the input, weights, biases and output tensor info.
     *
     * Valid data layouts:
     * - NDHWC
     *
     * Valid data type configurations:
     * |src0           |src1               |src2   |dst            |
     * |:--------------|:------------------|:------|:--------------|
     * |F16            |F16                |F16    |F16            |
     * |F32            |F32                |F32    |F32            |
     * |QASYMM8        |QASYMM8            |S32    |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32    |QASYMM8_SIGNED |
     *
     * @param[in, out] src0      Input tensor info.
     * @param[in]      src1      Set of kernels to convolve the input volume.
     *                           The 2nd dimension must be the same as the src0's volume 1st dimension.
     * @param[in]      src2      Set of biases. Can be nullptr.
     * @param[out]     dst       Output tensor info.
     *                           The 1st dimensions must be equal to the 1st dimension of the @p kernels tensor.
     * @param[in]      conv_info Contains padding, stride, acitvation information.
     */
    void configure(ITensorInfo *src0, ITensorInfo *src1, const ITensorInfo *src2, ITensorInfo *dst, const Conv3dInfo conv_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuDirectConv3d::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, const Conv3dInfo conv_info);

    // Inherited methods overridden:
    void run(ITensorPack &tensors) override;

private:
    MemoryGroup                                     _memory_group;
    std::unique_ptr<kernels::CpuDirectConv3dKernel> _conv_kernel;
    std::unique_ptr<CpuActivation>                  _activationlayer_function;
    Tensor                                          _accumulator;
    bool                                            _is_activationlayer_enabled{ false };
    unsigned int                                    _dim_split{ 0 };
};
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_DIRECTCONV3D_H */
