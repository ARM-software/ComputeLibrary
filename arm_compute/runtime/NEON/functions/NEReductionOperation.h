/*
 * Copyright (c) 2017-2020 ARM Limited.
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
#ifndef ARM_COMPUTE_NEREDUCTIONOPERATION_H
#define ARM_COMPUTE_NEREDUCTIONOPERATION_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/NEON/kernels/NEReductionOperationKernel.h"
#include "arm_compute/core/NEON/kernels/NEReshapeLayerKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"

namespace arm_compute
{
class ITensor;

/** Basic function to simulate a reduction operation. This function calls the following NEON kernels:
 *
 * -# @ref NEFillBorderKernel
 * -# @ref NEReductionOperationKernel
 *
 */
class NEReductionOperation : public IFunction
{
public:
    /** Default constructor */
    NEReductionOperation(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Set the input and output tensors.
     *
     * @param[in]  input     Source tensor. Data type supported: QASYMM8_SIGNED/QASYMM8/F16/F32. Data layouts supported: NCHW. (Written to only for border_size != 0)
     * @param[out] output    Destination tensor. Data types and data layouts supported: same as @p input.
     * @param[in]  axis      Dimension along which to reduce. Supported reduction axis : 0
     * @param[in]  op        Reduction operation to perform.
     * @param[in]  keep_dims (Optional) Whether to keep the reduced dimension after the operation. Defaults to true.
     */
    void configure(ITensor *input, ITensor *output, unsigned int axis, ReductionOperation op, bool keep_dims = true);

    /** Static function to check if given info will lead to a valid configuration of @ref NEReductionOperation.
     *
     * @param[in] input     Source tensor info. Data type supported: QASYMM8_SIGNED/QASYMM8/F16/F32. Data layouts supported: NCHW. (Written to only for border_size != 0)
     * @param[in] output    Destination tensor info. Data types and data layouts supported: same as @p input.
     * @param[in] axis      Dimension along which to reduce. Supported reduction axis : 0
     * @param[in] op        Reduction operation to perform.
     * @param[in] keep_dims (Optional) Whether to keep the reduced dimension after the operation. Defaults to true.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, unsigned int axis, ReductionOperation op, bool keep_dims = true);

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup                _memory_group;
    NEReductionOperationKernel _reduction_kernel;
    NEFillBorderKernel         _fill_border_kernel;
    NEReshapeLayerKernel       _reshape_kernel;
    Tensor                     _output_internal;
    size_t                     _window_split;
    int                        _reduction_axis;
    bool                       _is_reshape_required;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEREDUCTIONOPERATION_H */
