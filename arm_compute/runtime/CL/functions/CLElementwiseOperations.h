/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLELEMENTWISEOPERATIONS_H
#define ARM_COMPUTE_CLELEMENTWISEOPERATIONS_H

#include "arm_compute/runtime/CL/ICLOperator.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class ICLTensor;
class CLCompileContext;
class ITensorInfo;

/** Basic function to run @ref opencl::kernels::ClSaturatedArithmeticKernel for addition
 *
 * @note The tensor data type for the inputs must be U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
 * @note The function performs an arithmetic addition between two tensors.
 */
class CLArithmeticAddition : public IFunction
{
public:
    /** Default Constructor */
    CLArithmeticAddition();
    /** Default Destructor */
    ~CLArithmeticAddition();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLArithmeticAddition(const CLArithmeticAddition &) = delete;
    /** Default move constructor */
    CLArithmeticAddition(CLArithmeticAddition &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLArithmeticAddition &operator=(const CLArithmeticAddition &) = delete;
    /** Default move assignment operator */
    CLArithmeticAddition &operator=(CLArithmeticAddition &&);
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |src1           |dst            |
     * |:--------------|:--------------|:--------------|
     * |QASYMM8        |QASYMM8        |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |QASYMM8_SIGNED |
     * |QSYMM16        |QSYMM16        |QASYMM16       |
     * |U8             |U8             |U8             |
     * |U8             |U8             |S16            |
     * |U8             |S16            |S16            |
     * |S16            |U8             |S16            |
     * |S16            |S16            |S16            |
     * |S32            |S32            |S32            |
     * |F16            |F16            |F16            |
     * |F32            |F32            |F32            |
     *
     * @param[in, out] input1   First tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
     *                          The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2   Second tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
     *                          The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output   Output tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
     * @param[in]      policy   Policy to use to handle overflow.
     * @param[in]      act_info (Optional) Activation layer information in case of a fused activation.
     */
    void configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, ConvertPolicy policy, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * Valid configurations (Input1,Input2) -> Output :
     *
     *   - (U8,U8)           -> U8
     *   - (U8,U8)           -> S16
     *   - (S16,U8)          -> S16
     *   - (U8,S16)          -> S16
     *   - (S16,S16)         -> S16
     *   - (S32,S32)         -> S32
     *   - (F16,F16)         -> F16
     *   - (F32,F32)         -> F32
     *   - (QASYMM8,QASYMM8) -> QASYMM8
     *   - (QASYMM8_SIGNED,QASYMM8_SIGNED) -> QASYMM8_SIGNED
     *   - (QSYMM16,QSYMM16) -> QSYMM16
     *
     * @param[in]      compile_context The compile context to be used.
     * @param[in, out] input1          First tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
     *                                 The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2          Second tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
     *                                 The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output          Output tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
     * @param[in]      policy          Policy to use to handle overflow.
     * @param[in]      act_info        (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, ConvertPolicy policy,
                   const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref opencl::kernels::ClSaturatedArithmeticKernel for addition
     *
     * Valid configurations (Input1,Input2) -> Output :
     *
     *   - (U8,U8)           -> U8
     *   - (U8,U8)           -> S16
     *   - (S16,U8)          -> S16
     *   - (U8,S16)          -> S16
     *   - (S16,S16)         -> S16
     *   - (S32,S32)         -> S32
     *   - (F16,F16)         -> F16
     *   - (F32,F32)         -> F32
     *   - (QASYMM8,QASYMM8) -> QASYMM8
     *   - (QASYMM8_SIGNED,QASYMM8_SIGNED) -> QASYMM8_SIGNED
     *   - (QSYMM16,QSYMM16) -> QSYMM16
     *
     * @param[in] input1   First tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
     * @param[in] input2   Second tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
     * @param[in] output   Output tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
     * @param[in] policy   Policy to use to handle overflow.
     * @param[in] act_info (Optional) Activation layer information in case of a fused activation.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy, const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Basic function to run @ref opencl::kernels::ClSaturatedArithmeticKernel for subtraction
 *
 * @note The tensor data type for the inputs must be U8/QASYMM8/QASYMM8_SIGNED/S16/S32/F16/F32.
 * @note The function performs an arithmetic subtraction between two tensors.
 */
class CLArithmeticSubtraction : public IFunction
{
public:
    /** Default Constructor */
    CLArithmeticSubtraction();
    /** Default Destructor */
    ~CLArithmeticSubtraction();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLArithmeticSubtraction(const CLArithmeticSubtraction &) = delete;
    /** Default move constructor */
    CLArithmeticSubtraction(CLArithmeticSubtraction &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLArithmeticSubtraction &operator=(const CLArithmeticSubtraction &) = delete;
    /** Default move assignment operator */
    CLArithmeticSubtraction &operator=(CLArithmeticSubtraction &&);
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |src1           |dst            |
     * |:--------------|:--------------|:--------------|
     * |QASYMM8        |QASYMM8        |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |QASYMM8_SIGNED |
     * |QSYMM16        |QSYMM16        |QASYMM16       |
     * |U8             |U8             |U8             |
     * |U8             |U8             |S16            |
     * |U8             |S16            |S16            |
     * |S16            |U8             |S16            |
     * |S16            |S16            |S16            |
     * |S32            |S32            |S32            |
     * |F16            |F16            |F16            |
     * |F32            |F32            |F32            |
     *
     * @param[in, out] input1   First tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
     *                          The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2   Second tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
     *                          The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output   Output tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
     * @param[in]      policy   Policy to use to handle overflow.
     * @param[in]      act_info (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, ConvertPolicy policy, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * Valid configurations (Input1,Input2) -> Output :
     *
     *   - (U8,U8)           -> U8
     *   - (U8,U8)           -> S16
     *   - (S16,U8)          -> S16
     *   - (U8,S16)          -> S16
     *   - (S16,S16)         -> S16
     *   - (S32,S32)         -> S32
     *   - (F16,F16)         -> F16
     *   - (F32,F32)         -> F32
     *   - (QASYMM8,QASYMM8) -> QASYMM8
     *   - (QASYMM8_SIGNED,QASYMM8_SIGNED) -> QASYMM8_SIGNED
     *   - (QSYMM16,QSYMM16) -> QSYMM16
     *
     * @param[in]      compile_context The compile context to be used.
     * @param[in, out] input1          First tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
     *                                 The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2          Second tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
     *                                 The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output          Output tensor. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
     * @param[in]      policy          Policy to use to handle overflow.
     * @param[in]      act_info        (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, ConvertPolicy policy,
                   const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref opencl::kernels::ClSaturatedArithmeticKernel for subtraction
     *
     * Valid configurations (Input1,Input2) -> Output :
     *
     *   - (U8,U8)           -> U8
     *   - (U8,U8)           -> S16
     *   - (S16,U8)          -> S16
     *   - (U8,S16)          -> S16
     *   - (S16,S16)         -> S16
     *   - (S32,S32)         -> S32
     *   - (F16,F16)         -> F16
     *   - (F32,F32)         -> F32
     *   - (QASYMM8,QASYMM8) -> QASYMM8
     *   - (QASYMM8_SIGNED,QASYMM8_SIGNED) -> QASYMM8_SIGNED
     *   - (QSYMM16,QSYMM16) -> QSYMM16
     *
     * @param[in] input1   First tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
     * @param[in] input2   Second tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
     * @param[in] output   Output tensor info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/F16/F32.
     * @param[in] policy   Policy to use to handle overflow.
     * @param[in] act_info (Optional) Activation layer information in case of a fused activation.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy, const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Basic function to run @ref opencl::kernels::ClSaturatedArithmeticKernel for division
 *
 * @note The tensor data type for the inputs must be F16/F32.
 * @note The function performs an arithmetic division between two tensors.
 */
class CLArithmeticDivision : public IFunction
{
public:
    /** Default Constructor */
    CLArithmeticDivision();
    /** Default Destructor */
    ~CLArithmeticDivision();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLArithmeticDivision(const CLArithmeticDivision &) = delete;
    /** Default move constructor */
    CLArithmeticDivision(CLArithmeticDivision &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLArithmeticDivision &operator=(const CLArithmeticDivision &) = delete;
    /** Default move assignment operator */
    CLArithmeticDivision &operator=(CLArithmeticDivision &&);
    /** Initialise the kernel's inputs, output.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |src1           |dst            |
     * |:--------------|:--------------|:--------------|
     * |F16            |F16            |F16            |
     * |F32            |F32            |F32            |
     *
     * @param[in, out] input1   First tensor input. Data types supported: F16/F32.
     *                          The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2   Second tensor input. Same as @p input1.
     *                          The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output   Output tensor. Data types supported: Same as @p input1.
     * @param[in]      act_info (Optional) Activation layer information in case of a fused activation.
     */
    void configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Initialise the kernel's inputs, output.
     *
     * @param[in]      compile_context The compile context to be used.
     * @param[in, out] input1          First tensor input. Data types supported: F16/F32.
     *                                 The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2          Second tensor input. Same as @p input1.
     *                                 The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output          Output tensor. Data types supported: Same as @p input1.
     * @param[in]      act_info        (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CLArithmeticDivision
     *
     * @param[in] input1   First tensor input info. Data types supported: F16/F32.
     * @param[in] input2   Second tensor input info. Data types supported: Same as @p input1.
     * @param[in] output   Output tensor info. Data types supported: Same as @p input1.
     * @param[in] act_info (Optional) Activation layer information in case of a fused activation.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Basic function to run @ref opencl::kernels::ClArithmeticKernel for max
 *
 * @note The tensor data type for the inputs must be U8/QASYMM8/S16/QSYMM16/S32/U32/F16/F32.
 * @note The function performs a max operation between two tensors.
 */
class CLElementwiseMax : public IFunction
{
public:
    /** Default Constructor */
    CLElementwiseMax();
    /** Default Destructor */
    ~CLElementwiseMax();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLElementwiseMax(const CLElementwiseMax &) = delete;
    /** Default move constructor */
    CLElementwiseMax(CLElementwiseMax &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLElementwiseMax &operator=(const CLElementwiseMax &) = delete;
    /** Default move assignment operator */
    CLElementwiseMax &operator=(CLElementwiseMax &&);
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |src1           |dst            |
     * |:--------------|:--------------|:--------------|
     * |QASYMM8        |QASYMM8        |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |QASYMM8_SIGNED |
     * |QSYMM16        |QSYMM16        |QASYMM16       |
     * |U8             |U8             |U8             |
     * |S16            |S16            |S16            |
     * |S32            |S32            |S32            |
     * |U32            |U32            |U32            |
     * |F16            |F16            |F16            |
     * |F32            |F32            |F32            |
     *
     * @param[in, out] input1   First tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/U32/F16/F32.
     *                          The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2   Second tensor input. Data types supported: same as @p input1.
     *                          The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output   Output tensor. Data types supported: same as @p input1.
     * @param[in]      act_info (Optional) Activation layer information in case of a fused activation.
     */
    void configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in]      compile_context The compile context to be used.
     * @param[in, out] input1          First tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/U32/F16/F32.
     *                                 The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2          Second tensor input. Data types supported: same as @p input1.
     *                                 The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output          Output tensor. Data types supported: same as @p input1.
     * @param[in]      act_info        (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref opencl::kernels::ClArithmeticKernel for max
     *
     * @param[in] input1   First tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/U32/F16/F32.
     * @param[in] input2   Second tensor input info. Data types supported: same as @p input1.
     * @param[in] output   Output tensor info. Data types supported: same as @p input1.
     * @param[in] act_info (Optional) Activation layer information in case of a fused activation.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Basic function to run @ref opencl::kernels::ClArithmeticKernel for min
 *
 * @note The tensor data type for the inputs must be U8/QASYMM8/S16/QSYMM16/S32/U32/F16/F32.
 * @note The function performs a max operation between two tensors.
 */
class CLElementwiseMin : public IFunction
{
public:
    /** Default Constructor */
    CLElementwiseMin();
    /** Default Destructor */
    ~CLElementwiseMin();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLElementwiseMin(const CLElementwiseMin &) = delete;
    /** Default move constructor */
    CLElementwiseMin(CLElementwiseMin &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLElementwiseMin &operator=(const CLElementwiseMin &) = delete;
    /** Default move assignment operator */
    CLElementwiseMin &operator=(CLElementwiseMin &&);
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |src1           |dst            |
     * |:--------------|:--------------|:--------------|
     * |QASYMM8        |QASYMM8        |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |QASYMM8_SIGNED |
     * |QSYMM16        |QSYMM16        |QASYMM16       |
     * |U8             |U8             |U8             |
     * |S16            |S16            |S16            |
     * |S32            |S32            |S32            |
     * |U32            |U32            |U32            |
     * |F16            |F16            |F16            |
     * |F32            |F32            |F32            |
     *
     * @param[in, out] input1   First tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/U32/F16/F32.
     *                          The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2   Second tensor input. Data types supported: same as @p input1.
     *                          The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output   Output tensor. Data types supported: same as @p input1.
     * @param[in]      act_info (Optional) Activation layer information in case of a fused activation.
     */
    void configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in]      compile_context The compile context to be used.
     * @param[in, out] input1          First tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/U32/F16/F32.
     *                                 The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2          Second tensor input. Data types supported: same as @p input1.
     *                                 The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output          Output tensor. Data types supported: same as @p input1.
     * @param[in]      act_info        (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref opencl::kernels::ClArithmeticKernel for min
     *
     * @param[in] input1   First tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/S32/U32/F16/F32.
     * @param[in] input2   Second tensor input info. Data types supported: same as @p input1.
     * @param[in] output   Output tensor info. Data types supported: same as @p input1.
     * @param[in] act_info (Optional) Activation layer information in case of a fused activation.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Basic function to run @ref opencl::kernels::ClArithmeticKernel for squared difference
 *
 * @note The tensor data type for the inputs must be QASYMM8/U8/S16/QSYMM16/F16/F32.
 * @note The function performs a squared different operation between two tensors (i.e., out[i] = (in1[i] - in2[i])^2
 */
class CLElementwiseSquaredDiff : public IFunction
{
public:
    /** Default Constructor */
    CLElementwiseSquaredDiff();
    /** Default Destructor */
    ~CLElementwiseSquaredDiff();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLElementwiseSquaredDiff(const CLElementwiseSquaredDiff &) = delete;
    /** Default move constructor */
    CLElementwiseSquaredDiff(CLElementwiseSquaredDiff &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLElementwiseSquaredDiff &operator=(const CLElementwiseSquaredDiff &) = delete;
    /** Default move assignment operator */
    CLElementwiseSquaredDiff &operator=(CLElementwiseSquaredDiff &&);
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |src1           |dst            |
     * |:--------------|:--------------|:--------------|
     * |QASYMM8        |QASYMM8        |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |QASYMM8_SIGNED |
     * |QSYMM16        |QSYMM16        |QASYMM16       |
     * |U8             |U8             |U8             |
     * |S16            |S16            |S16            |
     * |F16            |F16            |F16            |
     * |F32            |F32            |F32            |
     *
     * @param[in, out] input1   First tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/F32.
     *                          The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2   Second tensor input. Data types supported: same as @p input1.
     *                          The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output   Output tensor. Data types supported: same as @p input1.
     * @param[in]      act_info (Optional) Activation layer information in case of a fused activation.
     */
    void configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in]      compile_context The compile context to be used.
     * @param[in, out] input1          First tensor input. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/F32.
     *                                 The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2          Second tensor input. Data types supported: same as @p input1.
     *                                 The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output          Output tensor. Data types supported: same as @p input1.
     * @param[in]      act_info        (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref opencl::kernels::ClArithmeticKernel for squared difference
     *
     * @param[in] input1   First tensor input info. Data types supported: U8/QASYMM8/QASYMM8_SIGNED/S16/QSYMM16/F16/F32.
     * @param[in] input2   Second tensor input info. Data types supported: same as @p input1.
     * @param[in] output   Output tensor info. Data types supported: same as @p input1.
     * @param[in] act_info (Optional) Activation layer information in case of a fused activation.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/** Basic function to run @ref opencl::kernels::ClArithmeticKernel for power
 *
 * @note The tensor data type for the inputs must be F16/F32.
 * @note The function performs an elementwise power of in1 to in2 (i.e., out[i] = in1[i] ^ in2[i])
 */
class CLElementwisePower : public IFunction
{
public:
    /** Default Constructor */
    CLElementwisePower();
    /** Default Destructor */
    ~CLElementwisePower();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLElementwisePower(const CLElementwisePower &) = delete;
    /** Default move constructor */
    CLElementwisePower(CLElementwisePower &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLElementwisePower &operator=(const CLElementwisePower &) = delete;
    /** Default move assignment operator */
    CLElementwisePower &operator=(CLElementwisePower &&);
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src0           |src1           |dst            |
     * |:--------------|:--------------|:--------------|
     * |F16            |F16            |F16            |
     * |F32            |F32            |F32            |
     *
     * @param[in, out] input1   First tensor input. Data types supported: F16/F32.
     *                          The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2   Second tensor input. Data types supported: F16/F32.
     *                          The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output   Output tensor. Data types supported:F16/F32.
     * @param[in]      act_info (Optional) Activation layer information in case of a fused activation.
     */
    void configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Initialise the kernel's inputs, output and conversion policy.
     *
     * @param[in]      compile_context The compile context to be used.
     * @param[in, out] input1          First tensor input. Data types supported: F16/F32.
     *                                 The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[in, out] input2          Second tensor input. Data types supported: F16/F32.
     *                                 The input tensor is [in, out] because its TensorInfo might be modified inside the kernel in case of broadcasting of dimension 0.
     * @param[out]     output          Output tensor. Data types supported:F16/F32.
     * @param[in]      act_info        (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref opencl::kernels::ClArithmeticKernel for power
     *
     * @param[in] input1   First tensor input info. Data types supported: F16/F32.
     * @param[in] input2   Second tensor input info. Data types supported: F16/F32.
     * @param[in] output   Output tensor info. Data types supported: F16/F32.
     * @param[in] act_info (Optional) Activation layer information in case of a fused activation.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLELEMENTWISEOPERATIONS_H */
