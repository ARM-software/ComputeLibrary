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
#ifndef ARM_COMPUTE_NEACCUMULATE_H
#define ARM_COMPUTE_NEACCUMULATE_H

#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"

#include <cstdint>

namespace arm_compute
{
class ITensor;

/** Basic function to run @ref NEAccumulateKernel */
class NEAccumulate : public INESimpleFunctionNoBorder
{
public:
    /** Default constructor */
    NEAccumulate() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEAccumulate(const NEAccumulate &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEAccumulate &operator=(const NEAccumulate &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEAccumulate(NEAccumulate &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEAccumulate &operator=(NEAccumulate &&) = delete;
    /** Default destructor */
    ~NEAccumulate();
    /** Set the input and accumulation tensors
     *
     * @param[in]  input  Source tensor. Data type supported: U8.
     * @param[out] output Destination tensor. Data type supported: S16.
     */
    void configure(const ITensor *input, ITensor *output);
};

/** Basic function to run @ref NEAccumulateWeightedKernel */
class NEAccumulateWeighted : public INESimpleFunctionNoBorder
{
public:
    /** Default constructor */
    NEAccumulateWeighted() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEAccumulateWeighted(const NEAccumulateWeighted &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEAccumulateWeighted &operator=(const NEAccumulateWeighted &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEAccumulateWeighted(NEAccumulateWeighted &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEAccumulateWeighted &operator=(NEAccumulateWeighted &&) = delete;
    /** Default destructor */
    ~NEAccumulateWeighted();
    /** Set the input and accumulation tensors, and the scale value
     *
     * @param[in]     input    Source tensor. Data type supported: U8.
     * @param[in]     alpha    The input scalar value with a value input the range of [0, 1.0]
     * @param[in,out] output   Accumulated tensor. Data type supported: U8.
     * @param[in]     use_fp16 (Optional) If true the FP16 kernels will be used. If false F32 kernels are used.
     */
    void configure(const ITensor *input, float alpha, ITensor *output, bool use_fp16 = false);
};

/** Basic function to run @ref NEAccumulateSquaredKernel */
class NEAccumulateSquared : public INESimpleFunctionNoBorder
{
public:
    /** Default constructor */
    NEAccumulateSquared() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEAccumulateSquared(const NEAccumulateSquared &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEAccumulateSquared &operator=(const NEAccumulateSquared &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEAccumulateSquared(NEAccumulateSquared &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEAccumulateSquared &operator=(NEAccumulateSquared &&) = delete;
    /** Default destructor */
    ~NEAccumulateSquared();
    /** Set the input and accumulation tensors and the shift value.
     *
     * @param[in]     input  Source tensor. Data type supported: U8.
     * @param[in]     shift  The input with a value input the range of [0, 15]
     * @param[in,out] output Accumulated tensor. Data type supported: S16.
     */
    void configure(const ITensor *input, uint32_t shift, ITensor *output);
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEACCUMULATE_H */
