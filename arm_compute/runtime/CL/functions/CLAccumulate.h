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
#ifndef ARM_COMPUTE_CLACCUMULATE_H
#define ARM_COMPUTE_CLACCUMULATE_H

#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

#include <cstdint>

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;

/** Basic function to run @ref CLAccumulateKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
*/
class CLAccumulate : public ICLSimpleFunction
{
public:
    /** Set the input and accumulation tensors.
     *
     * @param[in]  input Source tensor. Data types supported: U8.
     * @param[out] accum Destination tensor. Data types supported: S16.
     */
    void configure(const ICLTensor *input, ICLTensor *accum);
    /** Set the input and accumulation tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data types supported: U8.
     * @param[out] accum           Destination tensor. Data types supported: S16.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *accum);
};

/** Basic function to run @ref CLAccumulateWeightedKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
*/
class CLAccumulateWeighted : public ICLSimpleFunction
{
public:
    /** Set the input and accumulation tensors, and the scale value.
     *
     * @param[in]     input Source tensor. Data types supported: U8.
     * @param[in]     alpha The input scalar value with a value input the range of [0, 1.0]. Data types supported: F32.
     * @param[in,out] accum Accumulated tensor. Data types supported: U8.
     */
    void configure(const ICLTensor *input, float alpha, ICLTensor *accum);
    /** Set the input and accumulation tensors, and the scale value.
     *
     * @param[in]     compile_context The compile context to be used.
     * @param[in]     input           Source tensor. Data types supported: U8.
     * @param[in]     alpha           The input scalar value with a value input the range of [0, 1.0]. Data types supported: F32.
     * @param[in,out] accum           Accumulated tensor. Data types supported: U8.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, float alpha, ICLTensor *accum);
};

/** Basic function to run @ref CLAccumulateSquaredKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
*/
class CLAccumulateSquared : public ICLSimpleFunction
{
public:
    /** Set the input and accumulation tensors and the shift value.
     *
     * @param[in]     input Source tensor. Data types supported: U8.
     * @param[in]     shift The input with a value input the range of [0, 15]. Data types supported: U32.
     * @param[in,out] accum Accumulated tensor. Data types supported: S16.
     */
    void configure(const ICLTensor *input, uint32_t shift, ICLTensor *accum);
    /** Set the input and accumulation tensors and the shift value.
     *
     * @param[in]     compile_context The compile context to be used.
     * @param[in]     input           Source tensor. Data types supported: U8.
     * @param[in]     shift           The input with a value input the range of [0, 15]. Data types supported: U32.
     * @param[in,out] accum           Accumulated tensor. Data types supported: S16.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, uint32_t shift, ICLTensor *accum);
};
}
#endif /*ARM_COMPUTE_CLACCUMULATE_H */
