/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CL_WEIGHTSRESHAPE_KERNEL_H
#define ARM_COMPUTE_CL_WEIGHTSRESHAPE_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
/** OpenCL kernel to perform reshaping on the weights used by convolution and locally connected layer
 *
 * Rearranges each 3-dimensional kernel to a single row leading to a matrix with linearized kernels.
 * In combination with the @ref opencl::kernels::ClIm2ColKernel can transform a convolution to a matrix multiplication.
 *
 * For example assuming a 3D weight kernel of 3x3 dimensions and depth of 2 we have:
 * @f[
 * \left( \begin{array}{ccc}
 * a000 & a001 & a002 \\
 * a010 & a011 & a012 \\
 * a020 & a021 & a022 \\
 * \end{array} \right)
 * \left( \begin{array}{ccc}
 * a100 & a101 & a102 \\
 * a110 & a111 & a112 \\
 * a120 & a121 & a122 \\
 * \end{array} \right)
 * \rightarrow
 * \left( \begin{array}{ccccccccc}
 * a000 & a001 & a002 & a010 & a011 & a012 & a020 & a021 & a022 & a100 & a101 & a102 & a110 & a111 & a112 & a120 & a121 & a122 \\
 * \end{array} \right)
 * @f]
 */
class ClWeightsReshapeKernel : public IClKernel
{
public:
    ClWeightsReshapeKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClWeightsReshapeKernel);
    /** Set the input and output of the kernel.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             The input tensor info to convert. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM] if shared,
     *                             and 5D tensor with dimensions [kernel_x, kernel_y, IFM, OFM,  num_patches] if unshared. Data types supported: All
     * @param[in]  biases          The shared biases tensor info to append.  Bias is 1D tensor with dimensions [OFM] if shared and 2D tensor with
     *                             dimensions [OFM, num_patches] if unshared. Data types supported: F16/F32, for quantized types this must be nullptr.
     *                             @warning Appending biases to weights reshaped matrix is not supported for quantized asymmetric types.
     * @param[out] dst             The output tensor info. Should be a 2D Tensor if there are no groups and the weights are not shared; a 3D Tensor otherwise.
     *                             Data types supported: Same as @p input
     * @param[in]  num_groups      (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is only supported for NCHW data layout
     *                             Number of groups greater than one are only supported for NCHW data layout, and the number of weights must be a multiple of it.
     */
    void configure(const ClCompileContext &compile_context, const ITensorInfo *src, const ITensorInfo *biases, ITensorInfo *dst, unsigned int num_groups = 1);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to ClWeightsReshapeKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *biases, const ITensorInfo *dst, unsigned int num_groups = 1);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;
};
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif /*ARM_COMPUTE_CL_WEIGHTSRESHAPE_KERNEL_H */