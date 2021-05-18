/*
 * Copyright (c) 2017-2021 Arm Limited.
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

#include "arm_compute/runtime/NEON/functions/NEReshapeLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include <memory>

namespace arm_compute
{
class ITensor;
class NEReductionOperationKernel;

/** Basic function to simulate a reduction operation. This function calls the following kernels:
 *
 * -# @ref NEReshapeLayer
 * -# @ref NEReductionOperationKernel
 *
 */
class NEReductionOperation : public IFunction
{
public:
    /** Default constructor */
    NEReductionOperation(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEReductionOperation(const NEReductionOperation &) = delete;
    /** Default move constructor */
    NEReductionOperation(NEReductionOperation &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEReductionOperation &operator=(const NEReductionOperation &) = delete;
    /** Default move assignment operator */
    NEReductionOperation &operator=(NEReductionOperation &&) = default;
    /** Default destructor */
    ~NEReductionOperation();
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |QASYMM8        |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED |
     * |F16            |F16            |
     * |F32            |F32            |
     * |S32            |S32            |
     *
     * @param[in, out] input     Source tensor. Data type supported: QASYMM8_SIGNED/QASYMM8/F16/F32/S32. (Written to only for border_size != 0)
     * @param[out]     output    Destination tensor. Data types and data layouts supported: same as @p input.
     * @param[in]      axis      Dimension along which to reduce. Supported reduction axis : 0
     * @param[in]      op        Reduction operation to perform.
     * @param[in]      keep_dims (Optional) Whether to keep the reduced dimension after the operation. Defaults to true.
     */
    void configure(ITensor *input, ITensor *output, unsigned int axis, ReductionOperation op, bool keep_dims = true);

    /** Static function to check if given info will lead to a valid configuration of @ref NEReductionOperation.
     *
     * @param[in] input     Source tensor info. Data type supported: QASYMM8_SIGNED/QASYMM8/F16/F32/S32.
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
    MemoryGroup                                 _memory_group;
    std::unique_ptr<NEReductionOperationKernel> _reduction_kernel;
    NEReshapeLayer                              _reshape;
    Tensor                                      _output_internal;
    size_t                                      _window_split;
    int                                         _reduction_axis;
    bool                                        _is_reshape_required;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEREDUCTIONOPERATION_H */
