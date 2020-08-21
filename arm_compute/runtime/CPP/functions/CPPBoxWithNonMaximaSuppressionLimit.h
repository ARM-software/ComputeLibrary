/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CPPBOXWITHNONMAXIMASUPPRESSIONLIMIT_H
#define ARM_COMPUTE_CPPBOXWITHNONMAXIMASUPPRESSIONLIMIT_H

#include "arm_compute/core/CPP/kernels/CPPBoxWithNonMaximaSuppressionLimitKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

namespace arm_compute
{
class ITensor;

/** Basic function to run @ref CPPBoxWithNonMaximaSuppressionLimitKernel */
class CPPBoxWithNonMaximaSuppressionLimit : public IFunction
{
public:
    /** Constructor */
    CPPBoxWithNonMaximaSuppressionLimit(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPBoxWithNonMaximaSuppressionLimit(const CPPBoxWithNonMaximaSuppressionLimit &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPBoxWithNonMaximaSuppressionLimit &operator=(const CPPBoxWithNonMaximaSuppressionLimit &) = delete;
    /** Configure the BoxWithNonMaximaSuppressionLimit CPP kernel
     *
     * @param[in]  scores_in        The scores input tensor of size [count, num_classes]. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32
     * @param[in]  boxes_in         The boxes input tensor of size [count, num_classes * 4].
     *                              Data types supported: QASYMM16 with 0.125 scale and 0 offset if @p scores_in is QASYMM8/QASYMM8_SIGNED, otherwise same as @p scores_in
     * @param[in]  batch_splits_in  The batch splits input tensor of size [batch_size]. Data types supported: Same as @p scores_in
     *                              @note Can be a nullptr. If not a nullptr, @p scores_in and @p boxes_in have items from multiple images.
     * @param[out] scores_out       The scores output tensor of size [N]. Data types supported: Same as @p scores_in
     * @param[out] boxes_out        The boxes output tensor of size [N, 4].
     *                              Data types supported: QASYMM16 with 0.125 scale and 0 offset if @p scores_in is QASYMM8/QASYMM8_SIGNED, otherwise same as @p scores_in
     * @param[out] classes          The classes output tensor of size [N]. Data types supported: Same as @p scores_in
     * @param[out] batch_splits_out (Optional) The batch splits output tensor. Data types supported: Same as @p scores_in
     * @param[out] keeps            (Optional) The keeps output tensor of size [N]. Data types supported: Same as @p scores_in
     * @param[in]  keeps_size       (Optional) Number of filtered indices per class tensor of size [num_classes]. Data types supported: U32.
     * @param[in]  info             (Optional) BoxNMSLimitInfo information.
     */
    void configure(const ITensor *scores_in, const ITensor *boxes_in, const ITensor *batch_splits_in, ITensor *scores_out, ITensor *boxes_out, ITensor *classes,
                   ITensor *batch_splits_out = nullptr, ITensor *keeps = nullptr, ITensor *keeps_size = nullptr, const BoxNMSLimitInfo info = BoxNMSLimitInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CPPDetectionOutputLayer
     *
     * @param[in] scores_in        The scores input tensor of size [count, num_classes]. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32
     * @param[in] boxes_in         The boxes input tensor of size [count, num_classes * 4].
     *                             Data types supported: QASYMM16 with 0.125 scale and 0 offset if @p scores_in is QASYMM8/QASYMM8_SIGNED, otherwise same as @p scores_in
     * @param[in] batch_splits_in  The batch splits input tensor of size [batch_size]. Data types supported: Same as @p scores_in
     *                             @note Can be a nullptr. If not a nullptr, @p scores_in and @p boxes_in have items from multiple images.
     * @param[in] scores_out       The scores output tensor of size [N]. Data types supported: Same as @p scores_in
     * @param[in] boxes_out        The boxes output tensor of size [N, 4].
     *                             Data types supported: QASYMM16 with 0.125 scale and 0 offset if @p scores_in is QASYMM8/QASYMM8_SIGNED, otherwise same as @p scores_in
     * @param[in] classes          The classes output tensor of size [N]. Data types supported: Same as @p scores_in
     * @param[in] batch_splits_out (Optional) The batch splits output tensor. Data types supported: Same as @p scores_in
     * @param[in] keeps            (Optional) The keeps output tensor of size [N]. Data types supported: Same as @p scores_in
     * @param[in] keeps_size       (Optional) Number of filtered indices per class tensor of size [num_classes]. Data types supported: U32.
     * @param[in] info             (Optional) BoxNMSLimitInfo information.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *scores_in, const ITensorInfo *boxes_in, const ITensorInfo *batch_splits_in, const ITensorInfo *scores_out, const ITensorInfo *boxes_out,
                           const ITensorInfo *classes,
                           const ITensorInfo *batch_splits_out = nullptr, const ITensorInfo *keeps = nullptr, const ITensorInfo *keeps_size = nullptr, const BoxNMSLimitInfo info = BoxNMSLimitInfo());
    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup _memory_group;

    CPPBoxWithNonMaximaSuppressionLimitKernel _box_with_nms_limit_kernel;

    const ITensor *_scores_in;
    const ITensor *_boxes_in;
    const ITensor *_batch_splits_in;
    ITensor       *_scores_out;
    ITensor       *_boxes_out;
    ITensor       *_classes;
    ITensor       *_batch_splits_out;
    ITensor       *_keeps;

    Tensor _scores_in_f32;
    Tensor _boxes_in_f32;
    Tensor _batch_splits_in_f32;
    Tensor _scores_out_f32;
    Tensor _boxes_out_f32;
    Tensor _classes_f32;
    Tensor _batch_splits_out_f32;
    Tensor _keeps_f32;

    bool _is_qasymm8;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPPBOXWITHNONMAXIMASUPPRESSIONLIMIT_H */
