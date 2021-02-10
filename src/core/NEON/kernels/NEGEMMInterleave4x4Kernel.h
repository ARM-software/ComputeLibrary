/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NEGEMMINTERLEAVE4x4KERNEL_H
#define ARM_COMPUTE_NEGEMMINTERLEAVE4x4KERNEL_H

#include "src/core/NEON/INESimpleKernel.h"

namespace arm_compute
{
class ITensor;

/** Neon kernel to interleave the elements of a matrix
 *
 * This function puts the values in a 4x4 block of Matrix A on the same row (Interleaved values)
 *
 * @f[
 * \left( \begin{array}{cccc}
 * a00 & a01 & a02 & a03 \\
 * a10 & a11 & a12 & a13 \\
 * a20 & a21 & a22 & a23 \\
 * a30 & a31 & a32 & a33 \\
 * \end{array} \right)
 * \rightarrow
 * \left( \begin{array}{ccccccccccccccccc}
 * a00 & a10 & a20 & a30 & a01 & a11 & a21 & a31 & a02 & a12 & a22 & a32 & a03 & a13 & a23 & a33 \\
 * \end{array} \right)
 * @f]
 *
 * After this operation, the output matrix will have the following shape: [ height * 4, ceil(width / 4.0f) ]
 */
class NEGEMMInterleave4x4Kernel : public INESimpleKernel
{
public:
    const char *name() const override
    {
        return "NEGEMMInterleave4x4Kernel";
    }
    /** Constructor */
    NEGEMMInterleave4x4Kernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMInterleave4x4Kernel(const NEGEMMInterleave4x4Kernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMInterleave4x4Kernel &operator=(const NEGEMMInterleave4x4Kernel &) = delete;
    /** Allow instances of this class to be moved */
    NEGEMMInterleave4x4Kernel(NEGEMMInterleave4x4Kernel &&) = default;
    /** Allow instances of this class to be moved */
    NEGEMMInterleave4x4Kernel &operator=(NEGEMMInterleave4x4Kernel &&) = default;
    /** Default destructor */
    ~NEGEMMInterleave4x4Kernel() = default;
    /** Initialise the kernel's input and output.
     *
     * @param[in]  input  Input tensor. Data types supported: All
     * @param[out] output Output tensor which stores the interleaved matrix. Data type supported: same as @p input.
     */
    void configure(const ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMInterleave4x4Kernel
     *
     * @param[in] input  Input tensor info. Data types supported: All
     * @param[in] output Output tensor info which stores the interleaved matrix. Data type supported: same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Template function to run gemm interleave 4x4
     *
     * @tparam ScalarType Scalar datatype
     *
     * @param[in]  input  Input tensor. Data types supported: uint32_t, uint16_t and uint8_t
     * @param[out] output Output tensor. Data types supported: uint32_t, uint16_t and uint8_t
     * @param[in]  window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    template <typename ScalarType>
    void gemm_interleave4x4(const ITensor *input, ITensor *output, const Window &window);

    /** Common signature for all the specialised gemm interleave 4x4 functions
     *
     * @param[in]  input  Input tensor. Data types supported: uint32_t, uint16_t and uint8_t
     * @param[out] output Output tensor. Data types supported: uint32_t, uint16_t and uint8_t
     * @param[in]  window Region on which to execute the kernel. (Must be a valid region of the window returned by window()).
     */
    using GEMMInterleaveFunctionFuncPtr = void (NEGEMMInterleave4x4Kernel::*)(const ITensor *input, ITensor *output, const Window &window);

    GEMMInterleaveFunctionFuncPtr _func;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEGEMMINTERLEAVE4x4KERNEL_H*/
