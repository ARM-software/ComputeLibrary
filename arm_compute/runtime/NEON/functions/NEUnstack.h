/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEUNSTACK_H__
#define __ARM_COMPUTE_NEUNSTACK_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/runtime/NEON/functions/NEStridedSlice.h"

#include <memory>

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Basic function to unpack a rank-R tensor into rank-(R-1) tensors. This function calls the following functions:
 *
 * -# @ref NEStridedSlice
 */
class NEUnstack : public IFunction
{
public:
    /** Default constructor */
    NEUnstack();
    /** Set the input, output and unstacking axis.
     *
     * @param[in]     input         A tensor to be unstacked. Data type supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32.
     * @param[in,out] output_vector A vector of tensors. Data types supported: Same as @p input.
     *                              Note: The number of elements of the vector will be used as the number of slices to be taken from the axis.
     * @param[in]     axis          The axis to unstack along. Valid values are [-R,R) where R is the input's rank. Negative values wrap around.
     *
     */
    void configure(const ITensor *input, const std::vector<ITensor *> &output_vector, int axis);
    /** Static function to check if given info will lead to a valid configuration of @ref NEUnstack
     *
     * @param[in] input         Input tensor info. Data type supported: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32
     * @param[in] output_vector Vector of output tensors' info. Data types supported: Same as @p input.
     * @param[in] axis          The axis to unstack along. Valid values are [-R,R) where R is the input's rank. Negative values wrap around.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const std::vector<ITensorInfo *> &output_vector, int axis);

    // Inherited methods overridden:
    void run() override;

private:
    unsigned int                _num_slices;
    std::vector<NEStridedSlice> _strided_slice_vector;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEUNSTACK_H__ */
