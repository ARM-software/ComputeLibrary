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
#ifndef __ARM_COMPUTE_NEACCUMULATEKERNEL_H__
#define __ARM_COMPUTE_NEACCUMULATEKERNEL_H__

#include "arm_compute/core/NEON/INESimpleKernel.h"

#include <cstdint>

namespace arm_compute
{
class ITensor;

/** Interface for the accumulate kernel
 *
 * Accumulation is computed by:
 * @f[ accum(x,y) = accum(x,y) + input(x,y) @f]
 */
class NEAccumulateKernel : public INESimpleKernel
{
public:
    /** Set the input and accumulation tensors
     *
     * @param[in]  input Source tensor. Data type supported: U8.
     * @param[out] accum Destination tensor. Data type supported: S16.
     */
    void configure(const ITensor *input, ITensor *accum);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
};

/** Interface for the accumulate weighted kernel
 *
 * Weighted accumulation is computed:
 * @f[ accum(x,y) = (1 - \alpha)*accum(x,y) + \alpha*input(x,y) @f]
 *
 * Where @f$ 0 \le \alpha \le 1 @f$
 * Conceptually, the rounding for this is defined as:
 * @f[ output(x,y)= uint8( (1 - \alpha) * float32( int32( output(x,y) ) ) + \alpha * float32( int32( input(x,y) ) ) ) @f]
*/
class NEAccumulateWeightedKernel : public INESimpleKernel
{
public:
    /** Default constructor */
    NEAccumulateWeightedKernel();
    /** Set the input and accumulation tensors, and the scale value
     *
     * @param[in]     input Source tensor. Data type supported: U8.
     * @param[in]     alpha Scalar value in the range [0.0f, 1.0f]
     * @param[in,out] accum Accumulated tensor. Data type supported: U8.
     */
    void configure(const ITensor *input, float alpha, ITensor *accum);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

protected:
    float _alpha;
};

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
/** Interface for the accumulate weighted kernel using F16 */
class NEAccumulateWeightedFP16Kernel : public NEAccumulateWeightedKernel
{
public:
    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
};
#else  /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
using NEAccumulateWeightedFP16Kernel = NEAccumulateWeightedKernel;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

/** Interface for the accumulate squared kernel
 *
 * The accumulation of squares is computed:
 * @f[ accum(x,y) = saturate_{int16} ( (uint16) accum(x,y) + (((uint16)(input(x,y)^2)) >> (shift)) ) @f]
 *
 * Where @f$ 0 \le shift \le 15 @f$
*/
class NEAccumulateSquaredKernel : public INESimpleKernel
{
public:
    /** Default constructor */
    NEAccumulateSquaredKernel();
    /** Set the input and accumulation tensors and the shift value.
     *
     * @param[in]     input Source tensor. Data type supported: U8.
     * @param[in]     shift Shift value in the range of [0, 15]
     * @param[in,out] accum Accumulated tensor. Data type supported: S16.
     */
    void configure(const ITensor *input, uint32_t shift, ITensor *accum);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    uint32_t _shift;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEACCUMULATEKERNEL_H__ */
