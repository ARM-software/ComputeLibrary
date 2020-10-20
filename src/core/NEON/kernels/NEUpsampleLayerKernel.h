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
#ifndef ARM_COMPUTE_NEUPSAMPLELAYERKERNEL_H
#define ARM_COMPUTE_NEUPSAMPLELAYERKERNEL_H

#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the Upsample layer kernel.*/
class NEUpsampleLayerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEUpsampleLayerKernel";
    }
    /** Default constructor */
    NEUpsampleLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEUpsampleLayerKernel(const NEUpsampleLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEUpsampleLayerKernel &operator=(const NEUpsampleLayerKernel &) = delete;
    /** Default Move Constructor. */
    NEUpsampleLayerKernel(NEUpsampleLayerKernel &&) = default;
    /** Default move assignment operator */
    NEUpsampleLayerKernel &operator=(NEUpsampleLayerKernel &&) = default;
    /** Default destructor */
    ~NEUpsampleLayerKernel() = default;
    /** Set the input output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[out] output Destination tensor. Data types supported: same as @p input.
     * @param[in]  info   Contains stride information described in @ref Size2D.
     * @param[in]  policy Defines the policy to fill the intermediate pixels.
     *
     */
    void configure(const ITensor *input, ITensor *output, const Size2D &info, const InterpolationPolicy policy);
    /** Static function to check if given info will lead to a valid configuration of @ref NEUpsampleLayerKernel
     *
     * @param[in] input  Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] output Destination tensor info. Data types supported: same as @p input.
     * @param[in] info   Contains stride information described in @ref Size2D.
     * @param[in] policy Defines the policy to fill the intermediate pixels.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const Size2D &info, const InterpolationPolicy policy);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Function to run upsample layer (NCHW)
     *
     * @param[in] window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    template <typename T, int S>
    void upsample_nchw(const Window &window);
    /** Function to run upsample layer (NHWC)
     *
     * @param[in] window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    template <typename T, int S>
    void upsample_nhwc(const Window &window);

    using UpsampleFunctionPtr = void (NEUpsampleLayerKernel::*)(const Window &window);

private:
    UpsampleFunctionPtr _func;
    const ITensor      *_input;
    ITensor            *_output;
    Size2D              _info;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEUPSAMPLELAYERKERNEL_H */
