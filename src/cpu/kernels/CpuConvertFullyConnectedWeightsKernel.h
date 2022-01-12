/*
 * Copyright (c) 2018-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_CONVERT_FULLYCONNECTED_WEIGHTS_KERNEL_H
#define ARM_COMPUTE_CPU_CONVERT_FULLYCONNECTED_WEIGHTS_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Interface to convert the 2D Fully Connected weights from NCHW to NHWC or vice versa.
 *
 * @note This function can be applied to the 2D weights used by a Fully Connected layer if:
 *       - It follows a Convolution layer
 *       - The data layout used by the network does not match the one the model has been trained in.
 *
 * @note This function assumes the weights are already reshaped (transposed)
 */
class CpuConvertFullyConnectedWeightsKernel : public ICpuKernel<CpuConvertFullyConnectedWeightsKernel>
{
public:
    CpuConvertFullyConnectedWeightsKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuConvertFullyConnectedWeightsKernel);
    /** Set the src and dst tensor.
     *
     * @param[in] src                  Source weights tensor info to convert. Must be 2 dimensional. Data types supported: All.
     * @param[in] dst                  The converted weights tensor info. Shape and Data Type: Same as @p src.
     * @param[in] original_input_shape Shape of the original src tensor (the one entering fully connected layer).
     * @param[in] data_layout          The data layout the weights have been trained in.
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst, const TensorShape &original_input_shape, DataLayout data_layout);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuConvertFullyConnectedWeightsKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, const TensorShape &original_input_shape, DataLayout data_layout);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

private:
    unsigned int _factor1{ 0 }; /* equals to the number of elements per original src plane if @p data_layout == NCHW; its number of channels otherwise */
    unsigned int _factor2{ 0 }; /* equals to the number of elements per original src plane if @p data_layout == NHWC; its number of channels otherwise */
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_CONVERT_FULLYCONNECTED_WEIGHTS_KERNEL_H */