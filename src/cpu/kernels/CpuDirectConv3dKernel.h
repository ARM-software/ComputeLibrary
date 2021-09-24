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
#ifndef ARM_COMPUTE_CPU_DIRECT_CONV3D_KERNEL_H
#define ARM_COMPUTE_CPU_DIRECT_CONV3D_KERNEL_H

#include "arm_compute/runtime/FunctionDescriptors.h"
#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"
namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Interface for the kernel to perform 3D Direct Convolution Layer. */
class CpuDirectConv3dKernel : public ICpuKernel
{
public:
    CpuDirectConv3dKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuDirectConv3dKernel);
    /** Set the src, weights, and dst tensor info.
     *
     * Valid data layouts:
     * - NDHWC
     *
     * Valid data type configurations:
     * |src0           |src1               |src2   |dst            |
     * |:--------------|:------------------|:------|:--------------|
     * |F16            |F16                |F16    |F16            |
     * |F32            |F32                |F32    |F32            |
     *
     * @param[in, out] src       Input tensor info.
     * @param[in]      weights   Set of kernels to convolve the input volume.
     *                           The 2nd dimension must be the same as the input's volume 1st dimension.
     * @param[in]      biases    Set of biases. Can be nullptr.
     * @param[out]     dst       Output tensor info.
     *                           The 1st dimensions must be equal to the 1st dimension of the @p kernels tensor.
     * @param[in]      conv_info Contains padding, stride, acitvation information.
     *
     */
    void configure(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst, const Conv3dInfo &conv_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuDirectConv3dKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const Conv3dInfo &conv_info);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

private:
    /* Template function for convolution NDHWC */
    template <typename T>
    void convolve_ndhwc(const Window &window, const ITensor *src, const ITensor *weights, const ITensor *biases, ITensor *dst);

    Conv3dInfo _conv_info{};
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /*ARM_COMPUTE_CPU_DIRECTCONV3D_KERNEL_H */
