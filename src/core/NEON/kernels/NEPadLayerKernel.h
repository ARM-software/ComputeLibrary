/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NEPADLAYERKERNEL_H
#define ARM_COMPUTE_NEPADLAYERKERNEL_H

#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Basic kernel to pad the input tensor given padding information. */
class NEPadLayerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEPadLayerKernel";
    }
    /** Default constructor */
    NEPadLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPadLayerKernel(const NEPadLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPadLayerKernel &operator=(const NEPadLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEPadLayerKernel(NEPadLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEPadLayerKernel &operator=(NEPadLayerKernel &&) = default;
    /** Default destructor */
    ~NEPadLayerKernel() = default;

    /** Initialize the function
     *
     * @param[in]  input          Source tensor. Data types supported: All.
     * @param[out] output         Output tensor. Data type supported: same as @p input
     * @param[in]  padding        The padding for each spatial dimension of the input tensor. The pair padding[i]
     *                            specifies the front and the end padding in the i-th dimension.
     * @param[in]  constant_value (Optional) Constant value to be used for the padding
     * @param[in]  mode           (Optional) Controls whether the padding should be filled with @p constant_value using CONSTANT.
     *                           Only CONSTANT padding mode is currently supported
     */
    void configure(ITensor *input, ITensor *output, const PaddingList &padding, const PixelValue constant_value = PixelValue(), const PaddingMode mode = PaddingMode::CONSTANT);
    /**  Static function to check if given info will lead to a valid configuration of @ref NEPadLayer.
     *
     * @param[in] input          Source tensor info. Data types supported: All.
     * @param[in] output         Output tensor info. Data type supported: same as @p input
     * @param[in] padding        The padding for each spatial dimension of the input tensor. The pair padding[i]
     *                           specifies the front and the end padding in the i-th dimension.
     * @param[in] constant_value (Optional) Constant value to be used for the padding
     * @param[in] mode           (Optional) Controls whether the padding should be filled with @p constant_value using CONSTANT.
     *                           Only CONSTANT padding mode is currently supported
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const PaddingList &padding, const PixelValue constant_value = PixelValue(), const PaddingMode mode = PaddingMode::CONSTANT);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

    /** Return minimum workload size of the relevant kernel
     *
     * @param[in] platform     The CPU platform used to create the context.
     * @param[in] thread_count Number of threads in the execution.
     *
     * @return[out] small_network_mws          Minimum workload size for requsted configuration.
     */
    size_t get_mws(const CPUInfo &platform, size_t thread_count) const override;

private:
    /** Template function to run the padding function with constant padding
     *
     * @param[in] window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    template <typename T>
    void run_pad_constant(const Window &window);

    /** Function to run the padding function with constant padding for 3D input and 1D, 2D, 3D padding
     *
     * @param[in] window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    void run_pad_constant_uint8_3Dinput_3Dpad(const Window &window);

    /** Common signature for all the specialised permute functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using PadFunctionPtr = void (NEPadLayerKernel::*)(const Window &window);

    PadFunctionPtr _func;
    const ITensor *_input;
    ITensor       *_output;
    PaddingList    _padding;
    PixelValue     _constant_value;
    PaddingMode    _mode;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEPADLAYERKERNEL_H */
