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
#ifndef ARM_COMPUTE_CLPIXELWISEMULTIPLICATIONKERNEL_H
#define ARM_COMPUTE_CLPIXELWISEMULTIPLICATIONKERNEL_H

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
// Forward declarations
class ICLTensor;

/** Interface for the pixelwise multiplication kernel. */
class CLPixelWiseMultiplicationKernel : public ICLKernel
{
public:
    /** Default constructor.*/
    CLPixelWiseMultiplicationKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPixelWiseMultiplicationKernel(const CLPixelWiseMultiplicationKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLPixelWiseMultiplicationKernel &operator=(const CLPixelWiseMultiplicationKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLPixelWiseMultiplicationKernel(CLPixelWiseMultiplicationKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLPixelWiseMultiplicationKernel &operator=(CLPixelWiseMultiplicationKernel &&) = default;
    /** Initialise the kernel's input, output and border mode.
     *
     * Valid configurations (Input1,Input2) -> Output :
     *
     *   - (U8,U8)                         -> U8
     *   - (U8,U8)                         -> S16
     *   - (U8,S16)                        -> S16
     *   - (S16,U8)                        -> S16
     *   - (S16,S16)                       -> S16
     *   - (F16,F16)                       -> F16
     *   - (F32,F32)                       -> F32
     *   - (QASYMM8,QASYMM8)               -> QASYMM8
     *   - (QASYMM8_SIGNED,QASYMM8_SIGNED) -> QASYMM8_SIGNED
     *   - (QSYMM16,QSYMM16)               -> QSYMM16
     *   - (QSYMM16,QSYMM16)               -> S32
     *
     * @param[in]  input1          An input tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/F32.
     * @param[in]  input2          An input tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/F32.
     * @param[out] output          The output tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/F32.
     * @param[in]  scale           Scale to apply after multiplication.
     *                             Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15.
     * @param[in]  overflow_policy Overflow policy. Supported overflow policies: Wrap, Saturate
     * @param[in]  rounding_policy Rounding policy. Supported rounding modes: to zero, to nearest even.
     * @param[in]  act_info        (Optional) Activation layer information in case of a fused activation.
     */
    void configure(ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output, float scale,
                   ConvertPolicy overflow_policy, RoundingPolicy rounding_policy, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Initialise the kernel's input, output and border mode.
     *
     * Valid configurations (Input1,Input2) -> Output :
     *
     *   - (U8,U8)                         -> U8
     *   - (U8,U8)                         -> S16
     *   - (U8,S16)                        -> S16
     *   - (S16,U8)                        -> S16
     *   - (S16,S16)                       -> S16
     *   - (F16,F16)                       -> F16
     *   - (F32,F32)                       -> F32
     *   - (QASYMM8,QASYMM8)               -> QASYMM8
     *   - (QASYMM8_SIGNED,QASYMM8_SIGNED) -> QASYMM8_SIGNED
     *   - (QSYMM16,QSYMM16)               -> QSYMM16
     *   - (QSYMM16,QSYMM16)               -> S32
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input1          An input tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/F32.
     * @param[in]  input2          An input tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/F32.
     * @param[out] output          The output tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/F32.
     * @param[in]  scale           Scale to apply after multiplication.
     *                             Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15.
     * @param[in]  overflow_policy Overflow policy. Supported overflow policies: Wrap, Saturate
     * @param[in]  rounding_policy Rounding policy. Supported rounding modes: to zero, to nearest even.
     * @param[in]  act_info        (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const CLCompileContext &compile_context, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output, float scale,
                   ConvertPolicy overflow_policy, RoundingPolicy rounding_policy, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CLPixelWiseMultiplicationKernel
     *
     * Valid configurations (Input1,Input2) -> Output :
     *
     *   - (U8,U8)                         -> U8
     *   - (U8,U8)                         -> S16
     *   - (U8,S16)                        -> S16
     *   - (S16,U8)                        -> S16
     *   - (S16,S16)                       -> S16
     *   - (F16,F16)                       -> F16
     *   - (F32,F32)                       -> F32
     *   - (QASYMM8,QASYMM8)               -> QASYMM8
     *   - (QASYMM8_SIGNED,QASYMM8_SIGNED) -> QASYMM8_SIGNED
     *   - (QSYMM16,QSYMM16)               -> QSYMM16
     *   - (QSYMM16,QSYMM16)               -> S32
     *
     * @param[in] input1          An input tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/F32.
     * @param[in] input2          An input tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/F32.
     * @param[in] output          The output tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/F32.
     * @param[in] scale           Scale to apply after multiplication.
     *                            Scale must be positive and its value must be either 1/255 or 1/2^n where n is between 0 and 15.
     * @param[in] overflow_policy Overflow policy. Supported overflow policies: Wrap, Saturate
     * @param[in] rounding_policy Rounding policy. Supported rounding modes: to zero, to nearest even.
     * @param[in] act_info        (Optional) Activation layer information in case of a fused activation.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, float scale,
                           ConvertPolicy overflow_policy, RoundingPolicy rounding_policy, const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run_op(const InputTensorMap &inputs, const OutputTensorMap &outputs, const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

private:
    const ITensorInfo *_input1;
    const ITensorInfo *_input2;
    ITensorInfo       *_output;
};

/** Interface for the complex pixelwise multiplication kernel. */
class CLComplexPixelWiseMultiplicationKernel : public ICLKernel
{
public:
    /** Default constructor.*/
    CLComplexPixelWiseMultiplicationKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLComplexPixelWiseMultiplicationKernel(const CLComplexPixelWiseMultiplicationKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLComplexPixelWiseMultiplicationKernel &operator=(const CLComplexPixelWiseMultiplicationKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLComplexPixelWiseMultiplicationKernel(CLComplexPixelWiseMultiplicationKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLComplexPixelWiseMultiplicationKernel &operator=(CLComplexPixelWiseMultiplicationKernel &&) = default;
    /** Initialise the kernel's input, output and border mode.
     *
     * @param[in]  input1   An input tensor info. Data types supported: F32. Number of channels supported: 2.
     * @param[in]  input2   An input tensor info. Data types supported: same as @p input1. Number of channels supported: same as @p input1.
     * @param[out] output   The output tensor info. Data types supported: same as @p input1. Number of channels supported: same as @p input1.
     * @param[in]  act_info (Optional) Activation layer information in case of a fused activation.
     */
    void configure(ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Initialise the kernel's input, output and border mode.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input1          An input tensor info. Data types supported: F32. Number of channels supported: 2.
     * @param[in]  input2          An input tensor info. Data types supported: same as @p input1. Number of channels supported: same as @p input1.
     * @param[out] output          The output tensor info. Data types supported: same as @p input1. Number of channels supported: same as @p input1.
     * @param[in]  act_info        (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const CLCompileContext &compile_context, ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CLComplexPixelWiseMultiplicationKernel
     *
     * @param[in] input1   An input tensor info. Data types supported: F32. Number of channels supported: 2.
     * @param[in] input2   An input tensor info. Data types supported: same as @p input1. Number of channels supported: same as @p input1.
     * @param[in] output   The output tensor info. Data types supported: same as @p input1. Number of channels supported: same as @p input1.
     * @param[in] act_info (Optional) Activation layer information in case of a fused activation.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run_op(const InputTensorMap &inputs, const OutputTensorMap &outputs, const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

private:
    const ITensorInfo *_input1;
    const ITensorInfo *_input2;
    ITensorInfo       *_output;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLPIXELWISEMULTIPLICATIONKERNEL_H */
