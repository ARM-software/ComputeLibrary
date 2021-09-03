/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLCHANNELSHUFFLELAYER_H
#define ARM_COMPUTE_CLCHANNELSHUFFLELAYER_H

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to run @ref CLChannelShuffleLayerKernel
 *
 * @note The function performs a channel shuffle operation on the input tensor. Given NCHW tensor with group G, it will
 * first divide the channels into G groups, C = (G * C'), and perform a transpose of the channel, which gives C = (C' * G).
 * for more details see: https://arxiv.org/pdf/1707.01083.pdf
 */
class CLChannelShuffleLayer : public ICLSimpleFunction
{
public:
    /** Initialize the function
     *
     * Valid data layouts:
     * - NCHW
     * - NHWC
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |All            |All            |
     *
     * @param[in]  input      Input tensor. Data types supported: All.
     * @param[out] output     Output tensor. Data type supported: Same as @p input
     * @param[in]  num_groups Number of groups. Must be greater than 1 and the number of channels of the tensors must be a multiple of the number of groups.
     */
    void configure(const ICLTensor *input, ICLTensor *output, unsigned int num_groups);
    /** Initialize the function
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Input tensor. Data types supported: All.
     * @param[out] output          Output tensor. Data type supported: Same as @p input
     * @param[in]  num_groups      Number of groups. Must be greater than 1 and the number of channels of the tensors must be a multiple of the number of groups.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, unsigned int num_groups);
    /** Static function to check if given info will lead to a valid configuration of @ref CLChannelShuffleLayerKernel
     *
     * @param[in] input      Input tensor info. Data types supported: All.
     * @param[in] output     Output tensor info. Data type supported: Same as @p input
     * @param[in] num_groups Number of groups. Must be greater than 1 and the number of channels of the tensors must be a multiple of the number of groups.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, unsigned int num_groups);
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLCHANNELSHUFFLELAYER_H */
