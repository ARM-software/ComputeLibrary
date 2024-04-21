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
#ifndef ARM_COMPUTE_NECHANNELSHUFFLELAYERKERNEL_H
#define ARM_COMPUTE_NECHANNELSHUFFLELAYERKERNEL_H

#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Interface for the channel shuffle kernel */
class NEChannelShuffleLayerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEChannelShuffleLayerKernel";
    }
    /** Default constructor */
    NEChannelShuffleLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEChannelShuffleLayerKernel(const NEChannelShuffleLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEChannelShuffleLayerKernel &operator=(const NEChannelShuffleLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEChannelShuffleLayerKernel(NEChannelShuffleLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEChannelShuffleLayerKernel &operator=(NEChannelShuffleLayerKernel &&) = default;
    /** Default destructor */
    ~NEChannelShuffleLayerKernel() = default;
    /** Configure function's inputs and outputs.
     *
     * @param[in]  input      Input tensor. Data types supported: All
     * @param[out] output     Output tensor. Data type supported: Same as @p input
     * @param[in]  num_groups Number of groups. Must be greater than 1 and the number of channels of the tensors must be a multiple of the number of groups.
     */
    void configure(const ITensor *input, ITensor *output, unsigned int num_groups);
    /** Static function to check if given info will lead to a valid configuration of @ref NEChannelShuffleLayerKernel
     *
     * @param[in]  input      Input tensor. Data types supported: All
     * @param[out] output     Output tensor. Data type supported: Same as @p input
     * @param[in]  num_groups Number of groups. Must be greater than 1 and the number of channels of the tensors must be a multiple of the number of groups.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, unsigned int num_groups);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor *_input;
    ITensor       *_output;
    unsigned int   _num_groups;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NECHANNELSHUFFLELAYERKERNEL_H */
