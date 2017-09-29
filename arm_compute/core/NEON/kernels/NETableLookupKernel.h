/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_NETABLELOOKUPKERNEL_H__
#define __ARM_COMPUTE_NETABLELOOKUPKERNEL_H__

#include "arm_compute/core/NEON/INESimpleKernel.h"

namespace arm_compute
{
class ITensor;
class ILut;

/** Interface for the kernel to perform table lookup calculations. */
class NETableLookupKernel : public INESimpleKernel
{
public:
    /** Default constructor */
    NETableLookupKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NETableLookupKernel(const NETableLookupKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NETableLookupKernel &operator=(const NETableLookupKernel &) = delete;
    /** Allow instances of this class to be moved */
    NETableLookupKernel(NETableLookupKernel &&) = default;
    /** Allow instances of this class to be moved */
    NETableLookupKernel &operator=(NETableLookupKernel &&) = default;
    /** Initialise the kernel's input, lut and output.
     *
     * @param[in]  input  An input tensor. Data types supported: U8/S16.
     * @param[in]  lut    The input LUT.
     * @param[out] output The output tensor. Data types supported: same as @p input
     */
    void configure(const ITensor *input, const ILut *lut, ITensor *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Perform table lookup on a given window.
     *
     * @param window window Region on which to execute the kernel.
     */
    template <class T>
    void tableLookup(const Window &window);
    /** Common signature for all the specialised lut functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using TableLookupFunction = void (NETableLookupKernel::*)(const Window &window);
    /** Sub function to use for the particular tensor types passed to configure() */
    TableLookupFunction _func;
    const ILut         *_lut;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NETABLELOOKUPKERNEL_H__ */
