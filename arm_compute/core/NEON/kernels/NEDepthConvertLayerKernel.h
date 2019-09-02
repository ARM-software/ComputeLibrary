/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_DEPTHCONVERTKERNEL_H__
#define __ARM_COMPUTE_DEPTHCONVERTKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Depth conversion kernel
 *  This function ignores the scale and zeroPoint of quanized tensors, i.e. QASYMM8 input is treated as uint8 values.
 */
class NEDepthConvertLayerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEDepthConvertLayerKernel";
    }
    /** Default constructor*/
    NEDepthConvertLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDepthConvertLayerKernel(const NEDepthConvertLayerKernel &) = delete;
    /** Default move constructor */
    NEDepthConvertLayerKernel(NEDepthConvertLayerKernel &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDepthConvertLayerKernel &operator=(const NEDepthConvertLayerKernel &) = delete;
    /** Default move assignment operator */
    NEDepthConvertLayerKernel &operator=(NEDepthConvertLayerKernel &&) = default;
    /** Set the input and output of the kernel
     *
     * Valid conversions Input -> Output :
     *
     *   - QASYMM8 -> U16, S16, S32, F32, F16
     *   - U8 -> U16, S16, S32, F32, F16
     *   - U16 -> U8, U32
     *   - S16 -> U8, S32
     *   - F16 -> QASYMM8, F32, S32, U8
     *   - S32 -> QASYMM8, F16, F32, U8
     *   - F32 -> QASYMM8, F16, S32, U8
     *
     * @param[in]  input  The input tensor to convert. Data types supported: QASYMM8/U8/U16/S16/F16/F32.
     * @param[out] output The output tensor. Data types supported: QASYMM8/U8/U16/S16/U32/S32/F16/F32.
     * @param[in]  policy Conversion policy.
     * @param[in]  shift  (Optional) Value for down/up conversions. Must be 0 <= shift < 8.
     */
    void configure(const ITensor *input, ITensor *output, ConvertPolicy policy, uint32_t shift = 0);
    /** Static function to check if given info will lead to a valid configuration of @ref NEDepthConvertLayerKernel
     *
     * @param[in] input  Source tensor info. Data types supported: QASYMM8/U8/U16/S16/F16/F32.
     * @param[in] output Destination tensor info. Data type supported: QASYMM8/U8/U16/S16/U32/S32/F16/F32.
     * @param[in] policy Conversion policy
     * @param[in] shift  (Optional) Value for down/up conversions. Must be 0 <= shift < 8.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, ConvertPolicy policy, uint32_t shift = 0);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor *_input;
    ITensor       *_output;
    ConvertPolicy  _policy;
    uint32_t       _shift;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEDEPTHCONVERTKERNEL_H__ */
