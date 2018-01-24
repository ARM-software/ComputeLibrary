/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEPOOLINGLAYERKERNEL_H__
#define __ARM_COMPUTE_NEPOOLINGLAYERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the pooling layer kernel */
class NEPoolingLayerKernel : public INEKernel
{
public:
    /** Default constructor */
    NEPoolingLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPoolingLayerKernel(const NEPoolingLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPoolingLayerKernel &operator=(const NEPoolingLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEPoolingLayerKernel(NEPoolingLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEPoolingLayerKernel &operator=(NEPoolingLayerKernel &&) = default;
    /** Default destructor */
    ~NEPoolingLayerKernel() = default;
    /** Set the input and output tensors.
     *
     * @note QS8, QS16 and F16 are supported for pool sizes 2 and 3 only
     *
     * @param[in]  input     Source tensor. Data types supported: QS8/QASYMM8/QS16/F16/F32.
     * @param[out] output    Destination tensor. Data types supported: Same as @p input.
     * @param[in]  pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     */
    void configure(const ITensor *input, ITensor *output, const PoolingLayerInfo &pool_info);
    /** Static function to check if given info will lead to a valid configuration of @ref NEPoolingLayerKernel
     *
     * @note QS8, QS16 and F16 are supported for pool sizes 2 and 3 only
     *
     * @param[in] input     Source tensor. Data types supported: QS8/QASYMM8/QS16/F16/F32.
     * @param[in] output    Destination tensor. Data types supported: Same as @p input.
     * @param[in] pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    /** Function to perform 2x2 pooling.
     *
     * @param[in] window_input Input region on which to execute the kernel.
     * @param[in] window       Output region on which to execute the kernel.
     */
    template <PoolingType pooling_type, bool exclude_padding = false>
    void pooling2_f32(const Window &window_input, const Window &window);
    /** Function to perform 2x2 pooling for float16_t.
     *
     * @param[in] window_input Input region on which to execute the kernel.
     * @param[in] window       Output region on which to execute the kernel.
     */
    template <PoolingType pooling_type, bool exclude_padding = false>
    void pooling2_f16(const Window &window_input, const Window &window);
    /** Function to perform 2x2 pooling for 8bit fixed point.
     *
     * @param[in] window_input Input region on which to execute the kernel.
     * @param[in] window       Output region on which to execute the kernel.
     */
    template <PoolingType pooling_type>
    void pooling2_q8(const Window &window_input, const Window &window);
    /** Function to perform 2x2 pooling for 8bit asymmetric fixed point.
     *
     * @param[in] window_input Input region on which to execute the kernel.
     * @param[in] window       Output region on which to execute the kernel.
     */
    template <PoolingType pooling_type, bool exclude_padding = false>
    void pooling2_qasymm8(const Window &window_input, const Window &window);
    /** Function to perform 2x2 pooling for 16bit fixed point.
     *
     * @param[in] window_input Input region on which to execute the kernel.
     * @param[in] window       Output region on which to execute the kernel.
     */
    template <PoolingType pooling_type>
    void pooling2_q16(const Window &window_input, const Window &window);
    /** Function to perform 3x3 pooling.
     *
     * @param[in] window_input Input region on which to execute the kernel.
     * @param[in] window       Output region on which to execute the kernel.
     */
    template <PoolingType pooling_type, bool exclude_padding = false>
    void pooling3_f32(const Window &window_input, const Window &window);
    /** Function to perform 3x3 pooling.
     *
     * @param[in] window_input Input region on which to execute the kernel.
     * @param[in] window       Output region on which to execute the kernel.
     */
    template <PoolingType pooling_type, bool exclude_padding = false>
    void pooling3_f16(const Window &window_input, const Window &window);
    /** Function to perform 3x3 pooling for 8bit fixed point.
     *
     * @param[in] window_input Input region on which to execute the kernel.
     * @param[in] window       Output region on which to execute the kernel.
     */
    template <PoolingType pooling_type>
    void pooling3_q8(const Window &window_input, const Window &window);
    /** Function to perform 3x3 pooling for 8bit quantized fixed point.
     *
     * @param[in] window_input Input region on which to execute the kernel.
     * @param[in] window       Output region on which to execute the kernel.
     */
    template <PoolingType pooling_type, bool exclude_padding = false>
    void pooling3_qasymm8(const Window &window_input, const Window &window);
    /** Function to perform 3x3 pooling for 16bit fixed point.
     *
     * @param[in] window_input Input region on which to execute the kernel.
     * @param[in] window       Output region on which to execute the kernel.
     */
    template <PoolingType pooling_type>
    void pooling3_q16(const Window &window_input, const Window &window);
    /** Function to perform 7x7 pooling.
     *
     * @param[in] window_input Input region on which to execute the kernel.
     * @param[in] window       Output region on which to execute the kernel.
     */
    template <PoolingType pooling_type, bool exclude_padding = false>
    void pooling7_f32(const Window &window_input, const Window &window);
    /** Function to perform NxN pooling.
     *
     * @param[in] window_input Input region on which to execute the kernel.
     * @param[in] window       Output region on which to execute the kernel.
     */
    template <PoolingType pooling_type, bool exclude_padding = false>
    void poolingN_qasymm8(const Window &window_input, const Window &window);
    /** Function to perform NxN pooling.
     *
     * @param[in] window_input Input region on which to execute the kernel.
     * @param[in] window       Output region on which to execute the kernel.
     */
    template <PoolingType pooling_type, bool exclude_padding = false>
    void poolingN_f32(const Window &window_input, const Window &window);
    /** Common signature for all the specialised Pooling functions
     *
     * @param[in] window_input Input region on which to execute the kernel.
     * @param[in] window       Output region on which to execute the kernel.
     */
    using PoolingFunction = void (NEPoolingLayerKernel::*)(const Window &window_input, const Window &window);

private:
    PoolingFunction  _func;
    const ITensor   *_input;
    ITensor         *_output;
    PoolingLayerInfo _pool_info;
    unsigned int     _num_elems_processed_per_iteration;
    BorderSize       _border_size;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEPOOLINGLAYERKERNEL_H__ */
