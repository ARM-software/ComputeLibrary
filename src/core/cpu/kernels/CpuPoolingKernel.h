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
#ifndef ARM_COMPUTE_CPU_POOLING_KERNEL_H
#define ARM_COMPUTE_CPU_POOLING_KERNEL_H

#include "arm_compute/core/Types.h"
#include "src/core/common/Macros.h"
#include "src/core/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Interface for the pooling layer kernel */
class CpuPoolingKernel : public ICpuKernel
{
public:
    const char *name() const override
    {
        return "CpuPoolingKernel";
    }
    /** Default constructor */
    CpuPoolingKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuPoolingKernel);
    /** Configure kernel for a given list of arguments
     *
     * @note F16 are supported for pool sizes 2 and 3 only
     *
     * @param[in]  src       Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[out] dst       Destination tensor info. Data types supported: Same as @p src.
     * @param[in]  pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     * @param[out] indices   (optional) The indices of the maximal values. Data type supported: U32.
     */
    void configure(ITensorInfo *src, ITensorInfo *dst, const PoolingLayerInfo &pool_info, ITensorInfo *indices = nullptr);
    /** Static function to check if given info will lead to a valid configuration of @ref CpuPoolingKernel
     *
     * @note F16 are supported for pool sizes 2 and 3 only
     *
     * @param[in] src       Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] dst       Destination tensor info. Data types supported: Same as @p src.
     * @param[in] pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     * @param[in] indices   (optional) The indices of the maximal values. Data type supported: U32.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, const PoolingLayerInfo &pool_info, const ITensorInfo *indices = nullptr);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    /** Function to perform 2x2 pooling.
     *
     * @param[in] window_src      src region on which to execute the kernel.
     * @param[in] window          dst region on which to execute the kernel.
     * @param[in] pooling_type    Pooling operation to be computed.
     * @param[in] exclude_padding Flag to specify exclusion of padding from the operation.
     */
    void pooling2_f32_nchw(const Window &window_src, const Window &window, PoolingType pooling_type, bool exclude_padding = false);
    /** Function to perform 2x2 pooling and compute the pooling indices. The indices can be used for max unpool.
     *
     * @param[in] window_src src region on which to execute the kernel.
     * @param[in] window     dst region on which to execute the kernel.
     */
    void pooling2_f32_nhwc_maxpool_indices(const Window &window_src, const Window &window);
    /** Function to perform MxN pooling for 32-bit floating point values.
     *
     * @param[in] window_src      src region on which to execute the kernel.
     * @param[in] window          dst region on which to execute the kernel.
     * @param[in] pooling_type    Pooling operation to be computed.
     * @param[in] exclude_padding Flag to specify exclusion of padding from the operation.
     */
    void poolingMxN_f32_nchw(const Window &window_src, const Window &window, PoolingType pooling_type, bool exclude_padding = false);
    /** Function to perform MxN pooling for 32-bit floating point values (NHWC).
     *
     * @param[in] window_src      src region on which to execute the kernel.
     * @param[in] window          dst region on which to execute the kernel.
     * @param[in] pooling_type    Pooling operation to be computed.
     * @param[in] exclude_padding Flag to specify exclusion of padding from the operation.
     */
    void poolingMxN_f32_nhwc(const Window &window_src, const Window &window, PoolingType pooling_type, bool exclude_padding = false);
    /** Function to perform 7x7 pooling.
     *
     * @param[in] window_src      src region on which to execute the kernel.
     * @param[in] window          dst region on which to execute the kernel.
     * @param[in] pooling_type    Pooling operation to be computed.
     * @param[in] exclude_padding Flag to specify exclusion of padding from the operation.
     */
    void pooling7_f32_nchw(const Window &window_src, const Window &window, PoolingType pooling_type, bool exclude_padding = false);
    /** Function to perform 3x3 pooling.
     *
     * @param[in] window_src      src region on which to execute the kernel.
     * @param[in] window          dst region on which to execute the kernel.
     * @param[in] pooling_type    Pooling operation to be computed.
     * @param[in] exclude_padding Flag to specify exclusion of padding from the operation.
     */
    void pooling3_f32_nchw(const Window &window_src, const Window &window, PoolingType pooling_type, bool exclude_padding = false);
    /** Function to perform 2x2 pooling for float16_t.
     *
     * @param[in] window_src      src region on which to execute the kernel.
     * @param[in] window          dst region on which to execute the kernel.
     * @param[in] pooling_type    Pooling operation to be computed.
     * @param[in] exclude_padding Flag to specify exclusion of padding from the operation.
     */
    void pooling2_f16_nchw(const Window &window_src, const Window &window, PoolingType pooling_type, bool exclude_padding = false);
    /** Function to perform 2x2 pooling and compute the pooling indices for FP32/FP16. The indices can be used for max unpool.
     *
     * @param[in] window_src src region on which to execute the kernel.
     * @param[in] window     dst region on which to execute the kernel.
     */
    template <typename T>
    void pooling2_nchw_maxpool_indices(const Window &window_src, const Window &window);
    /** Function to perform 2x2 pooling and compute the pooling indices. The indices can be used for max unpool.
     *
     * @param[in] window_src src region on which to execute the kernel.
     * @param[in] window     dst region on which to execute the kernel.
     */
    void pooling2_f16_nhwc_maxpool_indices(const Window &window_src, const Window &window);
    /** Function to perform 3x3 pooling.
     *
     * @param[in] window_src      src region on which to execute the kernel.
     * @param[in] window          dst region on which to execute the kernel.
     * @param[in] pooling_type    Pooling operation to be computed.
     * @param[in] exclude_padding Flag to specify exclusion of padding from the operation.
     */
    void pooling3_f16_nchw(const Window &window_src, const Window &window, PoolingType pooling_type, bool exclude_padding = false);
    /** Function to perform MxN pooling for 16-bit floating point values.
     *
     * @param[in] window_src      src region on which to execute the kernel.
     * @param[in] window          dst region on which to execute the kernel.
     * @param[in] pooling_type    Pooling operation to be computed.
     * @param[in] exclude_padding Flag to specify exclusion of padding from the operation.
     */
    void poolingMxN_f16_nchw(const Window &window_src, const Window &window, PoolingType pooling_type, bool exclude_padding = false);
    /** Function to perform MxN pooling for 16-bit floating point values. (NHWC)
     *
     * @param[in] window_src      src region on which to execute the kernel.
     * @param[in] window          dst region on which to execute the kernel.
     * @param[in] pooling_type    Pooling operation to be computed.
     * @param[in] exclude_padding Flag to specify exclusion of padding from the operation.
     */
    void poolingMxN_f16_nhwc(const Window &window_src, const Window &window, PoolingType pooling_type, bool exclude_padding = false);
    /** Template function to perform 2x2 pooling for 8bit quantized fixed point. (NCHW)
     *
     * @param[in] window_src      src region on which to execute the kernel.
     * @param[in] window          dst region on which to execute the kernel.
     * @param[in] pooling_type    Pooling operation to be computed.
     * @param[in] exclude_padding Flag to specify exclusion of padding from the operation.
     */
    template <typename T>
    void pooling2_q8_nchw(const Window &window_src, const Window &window, PoolingType pooling_type, bool exclude_padding = false);
    /** Template function to perform 3x3 pooling for 8bit quantized fixed point. (NCHW)
     *
     * @param[in] window_src      src region on which to execute the kernel.
     * @param[in] window          dst region on which to execute the kernel.
     * @param[in] pooling_type    Pooling operation to be computed.
     * @param[in] exclude_padding Flag to specify exclusion of padding from the operation.
     */
    template <typename T>
    void pooling3_q8_nchw(const Window &window_src, const Window &window, PoolingType pooling_type, bool exclude_padding = false);
    /** Template function to perform MxN pooling for 8-bit quantized. (NCHW)
     *
     * @param[in] window_src      src region on which to execute the kernel.
     * @param[in] window          dst region on which to execute the kernel.
     * @param[in] pooling_type    Pooling operation to be computed.
     * @param[in] exclude_padding Flag to specify exclusion of padding from the operation.
     */
    template <typename T>
    void poolingMxN_q8_nchw(const Window &window_src, const Window &window, PoolingType pooling_type, bool exclude_padding = false);
    /** Template function to perform MxN pooling for 8-bit quantized. (NHWC)
     *
     * @param[in] window_src      src region on which to execute the kernel.
     * @param[in] window          dst region on which to execute the kernel.
     * @param[in] pooling_type    Pooling operation to be computed.
     * @param[in] exclude_padding Flag to specify exclusion of padding from the operation.
     */
    template <typename T>
    void poolingMxN_q8_nhwc(const Window &window_src, const Window &window, PoolingType pooling_type, bool exclude_padding = false);
    /** Common signature for all the specialised Pooling functions
     *
     * @param[in] window_src      src region on which to execute the kernel.
     * @param[in] window          dst region on which to execute the kernel.
     * @param[in] pooling_type    Pooling operation to be computed.
     * @param[in] exclude_padding Flag to specify exclusion of padding from the operation.
     */
    using PoolingFunction = void (CpuPoolingKernel::*)(const Window &window_src, const Window &window, PoolingType pooling_type, bool exclude_padding);

private:
    PoolingFunction  _func{ nullptr };
    const ITensor   *_src{ nullptr };
    ITensor         *_dst{ nullptr };
    ITensor         *_indices{ nullptr };
    PoolingLayerInfo _pool_info{};
    DataLayout       _data_layout{ DataLayout::UNKNOWN };
    unsigned int     _num_elems_processed_per_iteration{ 0 };
    BorderSize       _border_size{ 0 };
    bool             _is_square{ false };
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /*ARM_COMPUTE_CPU_POOLING_KERNEL_H */
