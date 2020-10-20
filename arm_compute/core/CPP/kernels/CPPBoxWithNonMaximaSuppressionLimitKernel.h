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
#ifndef ARM_COMPUTE_CPPBOXWITHNONMAXIMASUPPRESSIONLIMITKERNEL_H
#define ARM_COMPUTE_CPPBOXWITHNONMAXIMASUPPRESSIONLIMITKERNEL_H

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensor;

/** CPP kernel to perform computation of BoxWithNonMaximaSuppressionLimit */
class CPPBoxWithNonMaximaSuppressionLimitKernel : public ICPPKernel
{
public:
    const char *name() const override
    {
        return "CPPBoxWithNonMaximaSuppressionLimitKernel";
    }
    /** Default constructor */
    CPPBoxWithNonMaximaSuppressionLimitKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPBoxWithNonMaximaSuppressionLimitKernel(const CPPBoxWithNonMaximaSuppressionLimitKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CPPBoxWithNonMaximaSuppressionLimitKernel &operator=(const CPPBoxWithNonMaximaSuppressionLimitKernel &) = delete;
    /** Allow instances of this class to be moved */
    CPPBoxWithNonMaximaSuppressionLimitKernel(CPPBoxWithNonMaximaSuppressionLimitKernel &&) = default;
    /** Allow instances of this class to be moved */
    CPPBoxWithNonMaximaSuppressionLimitKernel &operator=(CPPBoxWithNonMaximaSuppressionLimitKernel &&) = default;
    /** Initialise the kernel's input and output tensors.
     *
     * @param[in]  scores_in        The scores input tensor of size [num_classes, count]. Data types supported: F16/F32
     * @param[in]  boxes_in         The boxes input tensor of size [num_classes * 4, count]. Data types supported: Same as @p scores_in
     * @param[in]  batch_splits_in  The batch splits input tensor of size [batch_size]. Data types supported: Same as @p scores_in
     *                              @note Can be a nullptr. If not a nullptr, @p scores_in and @p boxes_in have items from multiple images.
     * @param[out] scores_out       The scores output tensor of size [N]. Data types supported: Same as @p scores_in
     * @param[out] boxes_out        The boxes output tensor of size [4, N]. Data types supported: Same as @p scores_in
     * @param[out] classes          The classes output tensor of size [N]. Data types supported: Same as @p scores_in
     * @param[out] batch_splits_out (Optional) The batch splits output tensor [batch_size]. Data types supported: Same as @p scores_in
     * @param[out] keeps            (Optional) The keeps output tensor of size [N]. Data types supported: Same as@p scores_in
     * @param[out] keeps_size       (Optional) Number of filtered indices per class tensor of size [num_classes]. Data types supported: U32
     * @param[in]  info             (Optional) BoxNMSLimitInfo information.
     */
    void configure(const ITensor *scores_in, const ITensor *boxes_in, const ITensor *batch_splits_in, ITensor *scores_out, ITensor *boxes_out, ITensor *classes,
                   ITensor *batch_splits_out = nullptr, ITensor *keeps = nullptr, ITensor *keeps_size = nullptr, const BoxNMSLimitInfo info = BoxNMSLimitInfo());

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    bool is_parallelisable() const override;

    template <typename T>
    void run_nmslimit();

private:
    const ITensor *_scores_in;
    const ITensor *_boxes_in;
    const ITensor *_batch_splits_in;
    ITensor        *_scores_out;
    ITensor        *_boxes_out;
    ITensor        *_classes;
    ITensor        *_batch_splits_out;
    ITensor        *_keeps;
    ITensor        *_keeps_size;
    BoxNMSLimitInfo _info;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPPBOXWITHNONMAXIMASUPPRESSIONLIMITKERNEL_H */
