/*
 * Copyright (c) 2017-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPUWINOGRADCONV2DKERNEL_H
#define ARM_COMPUTE_CPUWINOGRADCONV2DKERNEL_H

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/Steps.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/Tensor.h"
#include "src/core/NEON/kernels/assembly/winograd.hpp"
#include "src/core/NEON/kernels/convolution/common/tensor.hpp"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
class CpuWinogradConv2dTransformInputKernel final : public ICpuKernel<CpuWinogradConv2dTransformInputKernel>
{
public:
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuWinogradConv2dTransformInputKernel(const CpuWinogradConv2dTransformInputKernel &) = delete;

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuWinogradConv2dTransformInputKernel &operator=(const CpuWinogradConv2dTransformInputKernel &) = delete;

    /**  Prevent instances of this class from being moved it contains references.*/
    CpuWinogradConv2dTransformInputKernel(CpuWinogradConv2dTransformInputKernel &&) = delete;

    /**  Prevent instances of this class from being moved it contains references.*/
    CpuWinogradConv2dTransformInputKernel &operator=(CpuWinogradConv2dTransformInputKernel &&) = delete;

    CpuWinogradConv2dTransformInputKernel(arm_conv::winograd::WinogradImpl &w_impl, arm_conv::ConvolutionArgs &_c_args, uint32_t nthreads);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;

    const char *name() const override
    {
        return "CpuWinogradConv2dTransformInputKernel";
    }

private:
    arm_conv::winograd::WinogradImpl &_winograd_impl;
    arm_conv::ConvolutionArgs        &_conv_args;
    uint32_t                          _nthreads;
};
class CpuWinogradConv2dTransformOutputKernel : public ICpuKernel<CpuWinogradConv2dTransformOutputKernel>
{
public:
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuWinogradConv2dTransformOutputKernel(const CpuWinogradConv2dTransformOutputKernel &) = delete;

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CpuWinogradConv2dTransformOutputKernel &operator=(const CpuWinogradConv2dTransformOutputKernel &) = delete;

    /**  Prevent instances of this class from being moved it contains references.*/
    CpuWinogradConv2dTransformOutputKernel(CpuWinogradConv2dTransformOutputKernel &&) = delete;

    /**  Prevent instances of this class from being moved it contains references.*/
    CpuWinogradConv2dTransformOutputKernel &operator=(CpuWinogradConv2dTransformOutputKernel &&) = delete;

    CpuWinogradConv2dTransformOutputKernel(arm_conv::winograd::WinogradImpl &w_impl, arm_conv::ConvolutionArgs &_c_args, uint32_t nthreads);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;

    const char *name() const override
    {
        return "CpuWinogradConv2dTransformOutputKernel";
    }

private:
    arm_conv::winograd::WinogradImpl &_winograd_impl;
    const arm_conv::ConvolutionArgs &_conv_args;
    uint32_t                          _nthreads;
};

} // namespace cpu
} // namespace arm_compute
#endif /*ARM_COMPUTE_CPUWINOGRADCONV2DKERNEL_H*/
