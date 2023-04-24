/*
 * Copyright (c) 2017-2023 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_IM2COL_KERNEL_H
#define ARM_COMPUTE_CPU_IM2COL_KERNEL_H

#include "arm_compute/core/Size2D.h"
#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
class ITensor;
namespace cpu
{
namespace kernels
{
/** Interface for the im2col reshape kernel.
 *
 * Rearranges image blocks into columns. It is used to strip out each convolution block to a single column.
 * It is used to transform a convolution to a plain matrix multiplication.
 *
 * For example taking into account the image below and assuming 3x3 image blocks with stride of 1 we have:
 *
 * @f[
 * \left( \begin{array}{cccc}
 * a00 & a01 & a02 & a03 \\
 * a10 & a11 & a12 & a13 \\
 * a20 & a21 & a22 & a23 \\
 * a30 & a31 & a32 & a33 \\
 * \end{array} \right)
 * \rightarrow
 * \left( \begin{array}{ccccccccc}
 * a00 & a01 & a02 & a10 & a11 & a12 & a20 & a21 & a22 \\
 * a01 & a02 & a03 & a11 & a12 & a13 & a21 & a22 & a23 \\
 * a10 & a11 & a12 & a20 & a21 & a22 & a30 & a31 & a32 \\
 * a11 & a12 & a13 & a21 & a22 & a23 & a31 & a32 & a33 \\
 * \end{array} \right)
 * @f]
 */
class CpuIm2ColKernel : public ICpuKernel<CpuIm2ColKernel>
{
public:
    /** Default constructor */
    CpuIm2ColKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuIm2ColKernel);
    /** Set the input and output of the kernel.
     *
     * @param[in]  src             The input tensor info to convert. 3 lower dimensions represent a single input [width, height, IFM],
     *                             while every optional dimension from 4 and above represent a batch of inputs.
     *                             Data types supported: QASYMM8/QASYMM8_SIGNED/BFLOAT16/F16/F32
     *                             Note: QASYMM8/QASYMM8_SIGNED works only for has_bias = false
     * @param[out] dst             The output tensor info. Data types supported: Same as @p input
     * @param[in]  kernel_dims     The kernel dimensions (width and height).
     * @param[in]  conv_info       Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  has_bias        In case biases are provided expands the matrix with 1.
     * @param[in]  dilation        (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  num_groups      (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is not supported
     * @param[in]  input_pad_right (Optional) When fast-math is selected, per element padding for the im2col matrix may be necessary
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst, const Size2D &kernel_dims, const PadStrideInfo &conv_info,
                   bool has_bias, const Size2D &dilation = Size2D(1U, 1U), unsigned int num_groups = 1, unsigned int input_pad_right = 0);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuIm2ColKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, const Size2D &kernel_dims, const PadStrideInfo &conv_info,
                           bool has_bias, const Size2D &dilation = Size2D(1U, 1U), unsigned int num_groups = 1, unsigned int input_pad_right = 0);

    // Inherited methods overridden:
    void        run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;
    /** Return minimum workload size of the relevant kernel
     *
     * @param[in] platform     The CPU platform used to create the context.
     * @param[in] thread_count Number of threads in the execution.
     *
     * @return[out] small_network_mws          Minimum workload size for requsted configuration.
     */
    size_t get_mws(const CPUInfo &platform, size_t thread_count) const override;

private:
    /** Template function to run im2col
     *
     * @param[in]  src    The input tensor info
     * @param[out] dst    The output tensor info
     * @param[in]  window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    template <typename T, bool has_pads, bool is_nchw>
    void run_im2col(const ITensor *src, ITensor *dst, const Window &window);

    /** Common signature for all the specialised im2col functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using Im2ColFunctionPtr = void (CpuIm2ColKernel::*)(const ITensor *src, ITensor *dst, const Window &window);

    Im2ColFunctionPtr                     _func{ nullptr };
    std::pair<unsigned int, unsigned int> _convolved_dims{};
    PadStrideInfo                         _conv_info{};
    unsigned int                          _kernel_width{ 0 };
    unsigned int                          _kernel_height{ 0 };
    unsigned int                          _input_pad_right{ 0 };
    bool                                  _has_bias{ false };
    Size2D                                _dilation{ 1U, 1U };
    DataLayout                            _data_layout{ DataLayout::UNKNOWN };
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /*ARM_COMPUTE_CPU_IM2COL_KERNEL_H */
