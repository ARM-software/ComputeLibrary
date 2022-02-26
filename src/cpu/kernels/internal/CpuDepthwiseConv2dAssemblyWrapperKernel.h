/*
 * Copyright (c) 2019-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_DEPTHWISE_CONV2D_ASSEMBLY_WRAPPER_KERNEL_H
#define ARM_COMPUTE_CPU_DEPTHWISE_CONV2D_ASSEMBLY_WRAPPER_KERNEL_H

#include "arm_compute/core/Types.h"
#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"
#include "src/cpu/kernels/CpuKernelSelectionTypes.h"

namespace arm_conv
{
namespace depthwise
{
// Forward declarations
class IDepthwiseCommon;
} // depthwise
} // arm_conv

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** This class is a wrapper for the depthwise convolution assembly kernels.  */
class CpuDepthwiseConv2dAssemblyWrapperKernel final : public ICpuKernel<CpuDepthwiseConv2dAssemblyWrapperKernel>
{
public:
    /** Default constructor */
    CpuDepthwiseConv2dAssemblyWrapperKernel();
    ~CpuDepthwiseConv2dAssemblyWrapperKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuDepthwiseConv2dAssemblyWrapperKernel);

    /** Initialise the kernel's src and dst.
     *
     * @param[in]  src      Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  weights  Weights tensor info. These are 3D tensors with shape [kernel_x, kernel_y, IFM].
     *                      Data type supported: same as @p src or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p src is QASYMM8/QASYMM8_SIGNED.
     * @param[in]  bias     Bias tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                      Data type supported: same as @p src, S32 when @p src is QASYMM8/QASYMM8_SIGNED.
     * @param[out] dst      Destination tensor info. Data type supported: same as @p input.
     * @param[in]  info     Depthwise convolution layer meta-data.
     * @param[in]  cpu_info CPU information needed to select the most appropriate kernel.
     */
    void configure(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *bias, ITensorInfo *dst, const ConvolutionInfo &info, const CPUInfo &cpu_info);

    /** Indicates whether or not this function can be used to process the given parameters.
     *
     * Similar to @ref CpuDepthwiseConv2dAssemblyWrapperKernel::configure()
     *
     * @return a status.
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *bias, const ITensorInfo *dst, const ConvolutionInfo &info);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

    /** Pack bias and weights in a storage space for the assembly kernel
     *
     * @param[in] parameters_ptr Pointer to storage space.
     * @param[in] bias_ptr       Pointer to bias buffer.
     * @param[in] weights_ptr    Pointer to weights buffer.
     * @param[in] ld_weights_col Columns displacement for the weights tensor.
     * @param[in] ld_weights_row Rows displacement for the weights tensor.
     */
    void pack_parameters(void *parameters_ptr, void *bias_ptr, void *weights_ptr, size_t ld_weights_col, size_t ld_weights_row);

    /** Get the amount of storage space required for the rearranged weights and bias.
     *
     * @return size of workspace
     */
    size_t get_storage_size() const;

    /** Get size of the workspace needed by the assembly kernel.
     *
     * @param[in] num_threads        Maximum number of threads that are going to be spawned.
     * @param[in] num_input_channels Number of channels of the input tensor.
     *
     * @return size of workspace
     */
    size_t get_working_size(unsigned int num_threads, unsigned int num_input_channels) const;

    /** Was the asm kernel successfully configured?
     *
     * @return True if the asm kernel is configured and ready to run
     */
    bool is_configured() const;

    /** Return minimum workload size of the relevant kernel
     *
     * @param[in] platform     The CPU platform used to create the context.
     * @param[in] thread_count Number of threads in the execution.
     *
     * @return[out] small_network_mws          Minimum workload size for requsted configuration.
     */
    size_t get_mws(const CPUInfo &platform, size_t thread_count) const override;

private:
    std::unique_ptr<arm_conv::depthwise::IDepthwiseCommon> _kernel_asm;
    std::vector<int32_t>                                   _multipliers{};
    std::vector<int32_t>                                   _left_shifts{};
    std::vector<int32_t>                                   _right_shifts{};
    std::string                                            _name{};
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_DEPTHWISE_CONV2D_ASSEMBLY_WRAPPER_KERNEL_H */
