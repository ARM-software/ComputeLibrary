/*
 * Copyright (c) 2016-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CL_CAST_KERNEL_H
#define ARM_COMPUTE_CL_CAST_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
/** Casts a given tensor to a new type
 *
 * @note When casting between quantized types the scale and zeroPoint are ignored
 */
class ClCastKernel : public IClKernel
{
public:
    ClCastKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClCastKernel);
    /** Set the src and dst of the kernel.
     *
     * Valid conversions src -> dst :
     *
     *   - QSYMM8_PER_CHANNEL -> QASYMM8 (ATTENTION: it is the user's responsibility to keep track of the quantization info in the TensorInfo meta-data)
     *   - U8  -> S8, U16, S16, U32, S32, F16, F32
     *   - S8  -> U8, U16, S16, U32, S32, F16, F32
     *   - U16 -> U8, S8, S16, U32, S32, F16, F32
     *   - S16 -> U8, S8, U16, U32, S32, F16, F32
     *   - U32 -> U8, S8, U16, S16, S32, F16, F32
     *   - S32 -> U8, S8, U16, S16, U32, F16, F32
     *   - F16 -> U8, S8, U16, S16, U32, F32
     *   - F32 -> U8, S8, U16, S16, U32, F16
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             The source tensor to convert. Data types supported: U8/S8/QSYMM8_PER_CHANNEL/U16/S16/U32/S32/F16/F32.
     * @param[out] dst             The destination tensor. Data types supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32.
     * @param[in]  policy          Conversion policy
     */
    void configure(const CLCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst, ConvertPolicy policy);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClCastKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, ConvertPolicy policy);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, ::cl::CommandQueue &queue) override;
};
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_CAST_KERNEL_H */
