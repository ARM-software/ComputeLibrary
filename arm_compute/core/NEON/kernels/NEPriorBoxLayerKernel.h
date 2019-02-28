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
#ifndef __ARM_COMPUTE_NEPRIORBOXLAYERKERNEL_H__
#define __ARM_COMPUTE_NEPRIORBOXLAYERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the kernel to calculate prior boxes */
class NEPriorBoxLayerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEPriorBoxLayerKernel";
    }
    /** Default constructor */
    NEPriorBoxLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPriorBoxLayerKernel(const NEPriorBoxLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPriorBoxLayerKernel &operator=(const NEPriorBoxLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEPriorBoxLayerKernel(NEPriorBoxLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEPriorBoxLayerKernel &operator=(NEPriorBoxLayerKernel &&) = default;
    /** Set the input and output tensors.
     *
     * @param[in]  input1 First source tensor. Data types supported: F32. Data layouts supported: NCHW/NHWC.
     * @param[in]  input2 Second source tensor. Data types and layouts supported: same as @p input1
     * @param[out] output Destination tensor. Output dimensions are [W * H * num_priors * 4, 2]. Data type supported: same as @p input
     * @param[in]  info   Prior box layer info.
     */
    void configure(const ITensor *input1, const ITensor *input2, ITensor *output, const PriorBoxLayerInfo &info);
    /** Static function to check if given info will lead to a valid configuration of @ref NEPriorBoxLayerKernel
     *
     * @param[in] input1 First source tensor info. Data types supported: F32. Data layouts supported: NCHW/NHWC.
     * @param[in] input2 Second source tensor info. Data types and layouts supported: same as @p input1
     * @param[in] output Destination tensor info. Output dimensions are [W * H * num_priors * 4, 2]. Data type supported: same as @p input
     * @param[in] info   Prior box layer info.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const PriorBoxLayerInfo &info);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Stores the coordinates of the calculated prior boxes.
     *
     * @param[out] out        Output pointer.
     * @param[in]  offset     Output offset to write to.
     * @param[in]  center_x   Center pixel value on x-axis.
     * @param[in]  center_y   Center pixel value on y-axis.
     * @param[in]  box_width  Prior box width.
     * @param[in]  box_height Prior box height.
     * @param[in]  width      Input width.
     * @param[in]  height     Input height.
     */
    void store_coordinates(float *out, const int offset, const float center_x, const float center_y, const float box_width, const float box_height, const int width, const int height);
    /** Function to calculate prior boxes.
     *
     * @param[in] window Input region on which to execute the kernel.
     */
    void calculate_prior_boxes(const Window &window);

    const ITensor    *_input1;
    const ITensor    *_input2;
    ITensor          *_output;
    PriorBoxLayerInfo _info;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEPRIORBOXLAYERKERNEL_H__ */
