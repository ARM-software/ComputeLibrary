/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_CPPBOXWITHNONMAXIMASUPPRESSIONLIMIT_H__
#define __ARM_COMPUTE_CPPBOXWITHNONMAXIMASUPPRESSIONLIMIT_H__

#include "arm_compute/runtime/CPP/ICPPSimpleFunction.h"

#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensor;

/** Basic function to run @ref CPPBoxWithNonMaximaSuppressionLimitKernel */
class CPPBoxWithNonMaximaSuppressionLimit : public ICPPSimpleFunction
{
public:
    /** Configure the BoxWithNonMaximaSuppressionLimit CPP kernel
     *
     * @param[in]  scores_in        The scores input tensor of size [count, num_classes]. Data types supported: F16/F32
     * @param[in]  boxes_in         The boxes input tensor of size [count, num_classes * 4]. Data types supported: Same as @p scores_in
     * @param[in]  batch_splits_in  The batch splits input tensor of size [batch_size]. Data types supported: Same as @p scores_in
     *                              @note Can be a nullptr. If not a nullptr, @p scores_in and @p boxes_in have items from multiple images.
     * @param[out] scores_out       The scores output tensor of size [N]. Data types supported: Same as @p scores_in
     * @param[out] boxes_out        The boxes output tensor of size [N, 4]. Data types supported: Same as @p scores_in
     * @param[out] classes          The classes output tensor of size [N]. Data types supported: Same as @p scores_in
     * @param[out] batch_splits_out (Optional) The batch splits output tensor. Data types supported: Same as @p scores_in
     * @param[out] keeps            (Optional) The keeps output tensor of size [N]. Data types supported: Same as @p scores_in
     * @param[in]  keeps_size       (Optional) Number of filtered indices per class tensor of size [num_classes]. Data types supported: Same as @p scores_in
     * @param[in]  info             (Optional) BoxNMSLimitInfo information.
     */
    void configure(const ITensor *scores_in, const ITensor *boxes_in, const ITensor *batch_splits_in, ITensor *scores_out, ITensor *boxes_out, ITensor *classes,
                   ITensor *batch_splits_out = nullptr, ITensor *keeps = nullptr, ITensor *keeps_size = nullptr, const BoxNMSLimitInfo info = BoxNMSLimitInfo());
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CPPBOXWITHNONMAXIMASUPPRESSIONLIMIT_H__ */
