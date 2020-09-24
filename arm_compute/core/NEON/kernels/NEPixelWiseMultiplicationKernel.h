/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_NEPIXELWISEMULTIPLICATIONKERNEL_H
#define ARM_COMPUTE_NEPIXELWISEMULTIPLICATIONKERNEL_H

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensor;

/** Interface for the kernel to perform addition between two tensors */
class NEPixelWiseMultiplicationKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEPixelWiseMultiplicationKernel";
    }
    /** Default constructor */
    NEPixelWiseMultiplicationKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPixelWiseMultiplicationKernel(const NEPixelWiseMultiplicationKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEPixelWiseMultiplicationKernel &operator=(const NEPixelWiseMultiplicationKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEPixelWiseMultiplicationKernel(NEPixelWiseMultiplicationKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEPixelWiseMultiplicationKernel &operator=(NEPixelWiseMultiplicationKernel &&) = default;
    /** Default destructor */
    ~NEPixelWiseMultiplicationKernel() = default;
    /** Initialise the kernel's input, output and border mode.
     *
     * Valid configurations (Input1,Input2) -> Output :
     *
     *                                                       Support: Broadcast? Scale=1/255?
     *   - (U8,U8)                         -> U8, S16                 N          Y
     *   - (U8,S16)                        -> S16                     N          Y
     *   - (S16,U8)                        -> S16                     N          Y
     *   - (S16,S16)                       -> S16                     N          Y
     *   - (S32,S32)                       -> S32                     Y          N
     *   - (F16,F16)                       -> F16                     N          Y
     *   - (F32,F32)                       -> F32                     Y          Y
     *   - (QASYMM8,QASYMM8)               -> QASYMM8                 Y          Y
     *   - (QASYMM8_SIGNED,QASYMM8_SIGNED) -> QASYMM8_SIGNED          Y          Y
     *   - (QSYMM16,QSYMM16)               -> QSYMM16, S32            N          Y
     *
     * @note For @p scale equal to 1/255 only round to nearest even (implemented as round half up) is supported.
     *       For all other scale values only round to zero (implemented as round towards minus infinity) is supported.
     *
     * @param[in]  input1          First input tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/QSYMM16/F16/F32
     * @param[in]  input2          Second input tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/QSYMM16/F16/F32
     * @param[out] output          Output tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/QSYMM16/F16/F32
     * @param[in]  scale           Scale to apply after multiplication.
     *                             Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15.
     *                             If both @p input1, @p input2 and @p output are of datatype S32, scale cannot be 1/255
     * @param[in]  overflow_policy Overflow policy. ConvertPolicy cannot be WRAP if any of the inputs is of quantized datatype
     * @param[in]  rounding_policy Rounding policy.
     */
    void configure(ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output, float scale, ConvertPolicy overflow_policy, RoundingPolicy rounding_policy);
    /** Static function to check if given info will lead to a valid configuration of @ref NEPixelWiseMultiplicationKernel
     *
     * Valid configurations (Input1,Input2) -> Output :
     *                                                       Support: Broadcast? Scale=1/255?
     *   - (U8,U8)                         -> U8, S16                 N          Y
     *   - (U8,S16)                        -> S16                     N          Y
     *   - (S16,U8)                        -> S16                     N          Y
     *   - (S16,S16)                       -> S16                     N          Y
     *   - (S32,S32)                       -> S32                     Y          N
     *   - (F16,F16)                       -> F16                     N          Y
     *   - (F32,F32)                       -> F32                     Y          Y
     *   - (QASYMM8,QASYMM8)               -> QASYMM8                 Y          Y
     *   - (QASYMM8_SIGNED,QASYMM8_SIGNED) -> QASYMM8_SIGNED          Y          Y
     *   - (QSYMM16,QSYMM16)               -> QSYMM16, S32            N          Y
     *
     * @note For @p scale equal to 1/255 only round to nearest even (implemented as round half up) is supported.
     *       For all other scale values only round to zero (implemented as round towards minus infinity) is supported.
     *
     * @param[in] input1          First input tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/QSYMM16/F16/F32
     * @param[in] input2          Second input tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/QSYMM16/F16/F32
     * @param[in] output          Output tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/S32/QSYMM16/F16/F32
     * @param[in] scale           Scale to apply after multiplication.
     *                            Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15.
     *                            If both @p input1, @p input2 and @p output are of datatype S32, scale cannot be 1/255
     * @param[in] overflow_policy Overflow policy. ConvertPolicy cannot be WRAP if any of the inputs is of quantized datatype
     * @param[in] rounding_policy Rounding policy.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, float scale, ConvertPolicy overflow_policy, RoundingPolicy rounding_policy);

    // Inherited methods overridden
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;

private:
    /** Common signature for all the specialised multiplication functions with integer scaling factor
     *
     * @param[in]  in1    Input1 tensor object.
     * @param[in]  in2    Input2 tensor object.
     * @param[out] out    Output tensor object.
     * @param[in]  window Region on which to execute the kernel
     * @param[in]  scale  Integer scale factor.
     */
    using MulFunctionInt = void(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window, int scale);
    /** Common signature for all the specialised multiplication functions with float scaling factor
     *
     * @param[in]  in1    Input1 tensor object.
     * @param[in]  in2    Input2 tensor object.
     * @param[out] out    Output tensor object.
     * @param[in]  window Region on which to execute the kernel
     * @param[in]  scale  Float scale factor.
     */
    using MulFunctionFloat = void(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window, float scale);
    /** Common signature for all the specialised QASYMM8 multiplication functions with float scaling factor
     *
     * @param[in]  in1    Input1 tensor object.
     * @param[in]  in2    Input2 tensor object.
     * @param[out] out    Output tensor object.
     * @param[in]  window Region on which to execute the kernel
     * @param[in]  scale  Float scale factor.
     *
     */
    using MulFunctionQuantized = void(const ITensor *in1, const ITensor *in2, ITensor *out, const Window &window, float scale);

    MulFunctionFloat     *_func_float;
    MulFunctionInt       *_func_int;
    MulFunctionQuantized *_func_quantized;

private:
    float _scale;
    int   _scale_exponent;
};

/** Interface for the complex pixelwise multiplication kernel. */
class NEComplexPixelWiseMultiplicationKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEComplexPixelWiseMultiplicationKernel";
    }
    /** Initialise the kernel's input, output and border mode.
     *
     * @param[in]  input1 An input tensor. Data types supported: F32. Number of channels supported: 2 (complex tensor).
     * @param[in]  input2 An input tensor. Data types supported: same as @p input1. Number of channels supported: same as @p input1.
     * @param[out] output The output tensor, Data types supported: same as @p input1.  Number of channels supported: same as @p input1.
     */
    void configure(ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEComplexPixelWiseMultiplicationKernel
     *
     * @param[in] input1 An input tensor info. Data types supported: F32. Number of channels supported: 2 (complex tensor).
     * @param[in] input2 An input tensor info. Data types supported: same as @p input1. Number of channels supported: same as @p input1.
     * @param[in] output The output tensor info. Data types supported: same as @p input1. Number of channels supported: same as @p input1.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
};

} // namespace arm_compute
#endif /*ARM_COMPUTE_NEPIXELWISEMULTIPLICATIONKERNEL_H */
