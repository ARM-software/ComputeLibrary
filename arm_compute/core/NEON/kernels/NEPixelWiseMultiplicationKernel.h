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
#ifndef __ARM_COMPUTE_NEPIXELWISEMULTIPLICATIONKERNEL_H__
#define __ARM_COMPUTE_NEPIXELWISEMULTIPLICATIONKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensor;

/** Interface for the kernel to perform addition between two tensors */
class NEPixelWiseMultiplicationKernel : public INEKernel
{
public:
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
     * @note For @p scale equal to 1/255 only round to nearest even (implemented as round half up) is supported.
     *       For all other scale values only round to zero (implemented as round towards minus infinity) is supported.
     *       For QS8/QS16 scale = 1 is the only supported value.
     *
     * @param[in]  input1          An input tensor. Data types supported: U8/QS8/QS16/S16/F16/F32
     * @param[in]  input2          An input tensor. Data types supported: U8, QS8 (only if @p input1 is QS8), QS16 (only if @p input1 is QS16), S16/F16 (only if @p input1 is F16), F32 (only if @p input1 is F32).
     * @param[out] output          The output tensor. Data types supported: U8 (Only if both inputs are U8), QS8 (only if both inputs are QS8), QS16 (only if both inputs are QS16), S16/F16 (only if @p input1 is F16), F32 (only if both inputs are F32).
     * @param[in]  scale           Scale to apply after multiplication.
     *                             Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15.
     * @param[in]  overflow_policy Overflow policy.
     * @param[in]  rounding_policy Rounding policy.
     */
    void configure(const ITensor *input1, const ITensor *input2, ITensor *output, float scale, ConvertPolicy overflow_policy, RoundingPolicy rounding_policy);
    /** Static function to check if given info will lead to a valid configuration of @ref NEPixelWiseMultiplicationKernel
     *
     * @note For @p scale equal to 1/255 only round to nearest even (implemented as round half up) is supported.
     *       For all other scale values only round to zero (implemented as round towards minus infinity) is supported.
     *       For QS8/QS16 scale = 1 is the only supported value.
     *
     * @param[in] input1          An input tensor. Data types supported: U8/QS8/QS16/S16/F16/F32
     * @param[in] input2          An input tensor. Data types supported: U8, QS8 (only if @p input1 is QS8), QS16 (only if @p input1 is QS16), S16/F16 (only if @p input1 is F16), F32 (only if @p input1 is F32).
     * @param[in] output          The output tensor. Data types supported: U8 (Only if both inputs are U8), QS8 (only if both inputs are QS8), QS16 (only if both inputs are QS16), S16/F16 (only if @p input1 is F16), F32 (only if both inputs are F32).
     * @param[in] scale           Scale to apply after multiplication.
     *                            Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15.
     * @param[in] overflow_policy Overflow policy.
     * @param[in] rounding_policy Rounding policy.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, float scale, ConvertPolicy overflow_policy, RoundingPolicy rounding_policy);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Common signature for all the specialised multiplication functions with integer scaling factor
     *
     * @param[in]  input1_ptr Pointer to the first input tensor.
     * @param[in]  input2_ptr Pointer to the second input tensor.
     * @param[out] output_ptr Pointer to the output tensor.
     */
    using MulFunctionInt = void(const void *__restrict input1_ptr, const void *__restrict input2_ptr, void *__restrict output_ptr, int scale);
    /** Common signature for all the specialised multiplication functions with fixed-point values
     *
     * @param[in]  input1_ptr           Pointer to the first input tensor.
     * @param[in]  input2_ptr           Pointer to the second input tensor.
     * @param[in]  scale                Scaling factor.
     * @param[in]  fixed_point_position Fixed-point position that expresses the number of bits for the fractional part of the number.
     * @param[out] output_ptr           Pointer to the output tensor.
     */
    using MulFunctionQInt = void(const void *__restrict input1_ptr, const void *__restrict input2_ptr, void *__restrict output_ptr, int scale, int fixed_point_position);
    /** Common signature for all the specialised multiplication functions with float scaling factor
     *
     * @param[in]  input1_ptr Pointer to the first input tensor.
     * @param[in]  input2_ptr Pointer to the second input tensor.
     * @param[out] output_ptr Pointer to the output tensor.
     */
    using MulFunctionFloat = void(const void *__restrict input1_ptr, const void *__restrict input2_ptr, void *__restrict output_ptr, float scale);

    MulFunctionFloat *_func_float;
    MulFunctionInt   *_func_int;
    MulFunctionQInt *_func_q_int;

private:
    const ITensor *_input1;
    const ITensor *_input2;
    ITensor       *_output;
    float          _scale;
    int            _scale_exponent;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEPIXELWISEMULTIPLICATIONKERNEL_H__ */
