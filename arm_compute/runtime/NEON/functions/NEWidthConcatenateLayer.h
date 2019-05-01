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
#ifndef __ARM_COMPUTE_NEWIDTHCONCATENATELAYER_H__
#define __ARM_COMPUTE_NEWIDTHCONCATENATELAYER_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/kernels/NEWidthConcatenateLayerKernel.h"

#include "arm_compute/core/utils/misc/Requires.h"

#include <memory>
#include <type_traits>
#include <vector>

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Basic function to execute concatenate tensors along x axis. This function calls the following kernel:
 *
 * -# @ref NEWidthConcatenateLayerKernel
 *
 * @deprecated This function is deprecated and will be removed in release 19.08
 */
class NEWidthConcatenateLayer : public IFunction
{
public:
    /** Default constructor */
    NEWidthConcatenateLayer();
    /** Initialise the kernel's inputs vector and output.
     *
     * @param[in]  inputs_vector The vectors containing all the tensors to concatenate. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     *                           Dimensions of all the inputs should match apart for the width which can differ.
     * @param[out] output        Output tensor. Data types supported: Same as @p input.
     *                           Output tensor dimensions are the same with the inputs from the second dimension and above.
     *                           The first dimension (width) is the sum of the input tensors' widths.
     */
    void configure(std::vector<ITensor *> inputs_vector, ITensor *output);
    void configure(std::vector<const ITensor *> inputs_vector, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEWidthConcatenateLayer
     *
     * @param[in] inputs_vector The vectors containing all the tensors to concatenate. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32.
     *                          Dimensions of all the inputs should match apart for the width which can differ.
     * @param[in] output        Output tensor. Data types supported: Same as @p input.
     *                          Output tensor dimensions are the same with the inputs from the second dimension and above.
     *                          The first dimension (width) is the sum of the input tensors' widths.
     *
     * @return a status
     */
    static Status validate(const std::vector<ITensorInfo *> &inputs_vector, const ITensorInfo *output);
    static Status validate(const std::vector<const ITensorInfo *> &inputs_vector, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    std::vector<NEWidthConcatenateLayerKernel> _concat_kernels_vector;
    unsigned int                               _num_inputs;
    template <typename TensorType, REQUIRES_TA(std::is_same<typename std::remove_cv<TensorType>::type, ITensor>::value)>
    void configure_internal(std::vector<TensorType *> &&inputs_vector, ITensor *output);
    template <typename TensorInfoType, REQUIRES_TA(std::is_same<typename std::remove_cv<TensorInfoType>::type, ITensorInfo>::value)>
    static Status validate_internal(const std::vector<TensorInfoType *> &inputs_vector, const ITensorInfo *output);
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEWIDTHCONCATENATELAYER_H__ */
